"""Train PitWallCNN on the Open-Pit Mine Object Detection annotations.

WHAT THIS MODEL ACTUALLY LEARNS
-------------------------------
The only human labels available in this repository are single-class
LabelMe rectangles (label "1") drawn around open-pit mining areas in
1024x1024 aerial scenes. There is no crack-severity label anywhere in
the dataset.

So this trains exactly the task the labels support and nothing more:

    binary patch classifier
      positive = a 224x224 crop taken from inside an annotated
                 open-pit mine region
      negative = a 224x224 crop from the same scene that does not
                 overlap any annotated region

It is NOT a crack detector, NOT a slope-failure predictor, and the
score it produces is NOT a probability of rockfall. Any claim beyond
"this patch looks like annotated open-pit mine terrain" is unsupported
by the labels and must not be made.

Negatives come from the same scenes as positives because every image in
the dataset contains at least one mine, so no true mine-free scenes
exist. That bounds what the metrics mean: the model separates pit
regions from their surroundings *within* mining scenes. It has never
been shown an image with no mine in it.

METHODOLOGY
-----------
* The train/val/test split is by SOURCE IMAGE, not by crop. Crops from
  one scene are heavily correlated (same sensor, lighting, terrain), so
  splitting by crop would leak and inflate every number reported here.
* Preprocessing is byte-identical to backend/cv_engine.py: resize to
  224x224, float32, divide by 255, no ImageNet normalisation. A
  mismatch here would silently degrade inference.
* The decision threshold is chosen on VALIDATION and then applied
  unchanged to the test set.

Outputs:
    models/drone_cnn_model.pth   state_dict for cv_engine.PitWallCNN
    models/cnn_metrics.json      held-out metrics + provenance
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from cv_engine import PitWallCNN  # noqa: E402  (needs the path insert above)

SEED = 1337
IMG = 224
MAX_POS = 900
MAX_NEG = 900
EPOCHS = 8
BATCH = 32
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET = REPO / "data" / "Open-Pit-Mine-Object-Detection-Dataset"
ANN_DIRS = [DATASET / "annotation"]
IMG_DIRS = [
    DATASET / "images",
    DATASET / "Open-Pit-Mine-Object-Detection-Dataset" / "images",
    REPO / "data" / "images",
    REPO / "data" / "extracted_pit_images",
]


def find_image(stem: str) -> Path | None:
    for d in IMG_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_pairs():
    """Pair each annotation file with its image. Skips the stub files."""
    pairs = []
    skipped_no_image = 0
    skipped_no_box = 0
    for ann_dir in ANN_DIRS:
        if not ann_dir.exists():
            continue
        for ap in sorted(ann_dir.glob("*.json")):
            try:
                doc = json.loads(ap.read_text())
            except Exception:
                continue
            boxes = []
            for s in doc.get("shapes", []):
                if s.get("shape_type") != "rectangle":
                    continue
                pts = s.get("points") or []
                if len(pts) != 2:
                    continue
                (x1, y1), (x2, y2) = pts
                x1, x2 = sorted((float(x1), float(x2)))
                y1, y2 = sorted((float(y1), float(y2)))
                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue
                boxes.append((x1, y1, x2, y2))
            if not boxes:
                skipped_no_box += 1
                continue
            ip = find_image(ap.stem)
            if ip is None:
                skipped_no_image += 1
                continue
            pairs.append((ap.stem, ip, boxes))
    print(f"[data] usable image/annotation pairs : {len(pairs)}")
    print(f"[data] skipped, no matching image    : {skipped_no_image}")
    print(f"[data] skipped, no usable box        : {skipped_no_box}")
    return pairs


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def crop_to_array(im: Image.Image, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    patch = im.crop((x1, y1, x2, y2)).resize((IMG, IMG))
    return np.asarray(patch, dtype=np.uint8)


def build_crops(pairs):
    """Positives from inside boxes, negatives from non-overlapping windows."""
    rng = random.Random(SEED)
    rows = []  # (stem, array, label)
    pos = neg = 0
    for stem, ip, boxes in pairs:
        if pos >= MAX_POS and neg >= MAX_NEG:
            break
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            continue
        W, H = im.size

        if pos < MAX_POS:
            for b in boxes[:2]:
                rows.append((stem, crop_to_array(im, b), 1))
                pos += 1
                if pos >= MAX_POS:
                    break

        if neg < MAX_NEG:
            got = 0
            for _ in range(40):
                if got >= 2 or neg >= MAX_NEG:
                    break
                side = rng.randint(160, min(320, max(161, min(W, H))))
                if W - side <= 0 or H - side <= 0:
                    continue
                x = rng.randint(0, W - side)
                y = rng.randint(0, H - side)
                cand = (x, y, x + side, y + side)
                # Strictly disjoint from every annotated mine region.
                if max((iou(cand, b) for b in boxes), default=0.0) > 0.02:
                    continue
                rows.append((stem, crop_to_array(im, cand), 0))
                neg += 1
                got += 1
    print(f"[data] crops: {pos} positive, {neg} negative")
    return rows


def split_by_image(rows):
    """Group-wise split so no scene appears in two splits."""
    stems = sorted({r[0] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(stems)
    n = len(stems)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    tr = set(stems[:n_tr])
    va = set(stems[n_tr:n_tr + n_va])
    te = set(stems[n_tr + n_va:])
    out = {"train": [], "val": [], "test": []}
    for stem, arr, y in rows:
        key = "train" if stem in tr else ("val" if stem in va else "test")
        out[key].append((arr, y))
    for k, v in out.items():
        ones = sum(1 for _, y in v if y == 1)
        print(f"[split] {k:5s}: {len(v):5d} crops ({ones} pos / {len(v)-ones} neg)"
              f" from {len(tr if k=='train' else va if k=='val' else te)} scenes")
    return out


def to_tensors(items):
    """Keep crops as uint8 NHWC and convert per batch.

    Materialising every crop as float32 NCHW up front costs
    n * 224 * 224 * 3 * 4 bytes -- over a gigabyte for a few thousand
    crops, which is a bad way to treat someone's laptop. uint8 is a
    quarter of that and the per-batch conversion is free in comparison
    to the convolutions.
    """
    if not items:
        return torch.empty(0, dtype=torch.uint8), torch.empty(0)
    X = torch.from_numpy(np.stack([a for a, _ in items]))  # N,H,W,C uint8
    y = torch.tensor([[float(l)] for _, l in items], dtype=torch.float32)
    return X, y


def as_input(batch_u8):
    """uint8 NHWC -> float NCHW scaled by 1/255, exactly as cv_engine does.

    .contiguous() matters: permute only rewrites strides, and the
    non-contiguous view propagates through the conv stack to the
    flatten in PitWallCNN.forward, where .view() refuses it.
    """
    return batch_u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


@torch.no_grad()
def predict(model, X, batch=64):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        outs.append(model(as_input(X[i:i + batch])))
    return torch.cat(outs).squeeze(1).numpy() if outs else np.array([])


def binary_metrics(y_true, scores, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    return {"threshold": round(float(thr), 3), "accuracy": round(acc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def roc_auc(y_true, scores):
    """Rank-based AUC; ties get averaged ranks. No sklearn dependency."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[y_true == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    pairs = load_pairs()
    if len(pairs) < 50:
        print("\n[abort] fewer than 50 usable pairs; refusing to train a model "
              "nobody could trust. Check that the dataset images are extracted.")
        sys.exit(2)

    rows = build_crops(pairs)
    splits = split_by_image(rows)
    Xtr, ytr = to_tensors(splits["train"])
    Xva, yva = to_tensors(splits["val"])
    Xte, yte = to_tensors(splits["test"])
    if len(Xte) == 0 or len(Xva) == 0:
        print("[abort] empty val/test split")
        sys.exit(2)

    model = PitWallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCELoss()

    best_auc, best_state = -1.0, None
    n = len(Xtr)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(as_input(Xtr[idx]))
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        va_scores = predict(model, Xva)
        va_auc = roc_auc(yva.squeeze(1).numpy(), va_scores)
        print(f"[train] epoch {ep}/{EPOCHS}  loss={total/max(1,n):.4f}  val_auc={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Threshold picked on validation, then frozen for the test set.
    va_scores = predict(model, Xva)
    yva_np = yva.squeeze(1).numpy()
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        m = binary_metrics(yva_np, va_scores, thr)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], float(thr)

    te_scores = predict(model, Xte)
    yte_np = yte.squeeze(1).numpy()
    test = binary_metrics(yte_np, te_scores, best_thr)
    test_at_half = binary_metrics(yte_np, te_scores, 0.5)
    test_auc = roc_auc(yte_np, te_scores)

    models_dir = REPO / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "drone_cnn_model.pth")

    metrics = {
        "task": "binary patch classification: annotated open-pit mine region vs "
                "non-overlapping background crop from the same aerial scene",
        "not_a_claim_about": [
            "crack detection or crack severity",
            "slope failure or rockfall probability",
            "any image lacking an open-pit mine, which the model has never seen",
        ],
        "label_source": "Open-Pit Mine Object Detection Dataset, LabelMe rectangles, "
                        "single class '1'",
        "split": "grouped by source image (no scene appears in two splits)",
        "seed": SEED,
        "epochs": EPOCHS,
        "selected_on": "validation ROC-AUC",
        "val_roc_auc": round(float(best_auc), 4),
        "test_roc_auc": round(float(test_auc), 4),
        "test_at_tuned_threshold": test,
        "test_at_0.5": test_at_half,
        "n_train_crops": int(len(Xtr)),
        "n_val_crops": int(len(Xva)),
        "n_test_crops": int(len(Xte)),
    }
    (models_dir / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== held-out test set ===")
    print(f"  ROC-AUC            {test_auc:.4f}")
    print(f"  threshold (val)    {best_thr:.2f}")
    print(f"  accuracy           {test['accuracy']:.4f}")
    print(f"  precision          {test['precision']:.4f}")
    print(f"  recall             {test['recall']:.4f}")
    print(f"  f1                 {test['f1']:.4f}")
    print(f"  confusion          {test['confusion']}")
    print(f"\nwrote {models_dir/'drone_cnn_model.pth'}")
    print(f"wrote {models_dir/'cnn_metrics.json'}")
    print("\nThis model separates annotated pit regions from their surroundings.")
    print("It is not a crack detector. Do not describe it as one.")


if __name__ == "__main__":
    main()
