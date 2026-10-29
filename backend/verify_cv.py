"""Verification for the drone-image CNN path.

Follows the same contract as the other verify_*.py scripts: run it
directly, every check prints PASS/FAIL, and the process exits non-zero
if anything fails.

Covers the parts that are easy to get wrong and impossible to notice:
  * the trained weights actually load into PitWallCNN (a silent failure
    here leaves a randomly-initialised net answering questions)
  * the operating point is read from cnn_metrics.json, not hardcoded
  * forward passes work BATCHED, which is what regressed when
    PitWallCNN.forward used .view() on a permuted tensor
  * the response reports mine-terrain confidence and carries the caveat,
    and no longer claims anything about cracks
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import cv_engine  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def sample_image_bytes() -> bytes:
    """Prefer a real pit image; fall back to a synthetic one."""
    for d in (REPO / "data" / "extracted_pit_images",
              REPO / "data" / "Open-Pit-Mine-Object-Detection-Dataset" / "images"):
        if d.is_dir():
            for p in sorted(d.glob("*.jpg"))[:1]:
                return p.read_bytes(), p.name
    img = Image.fromarray(
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue(), "<synthetic>"


def main() -> int:
    print("\n=== 1. weights ===")
    wpath = REPO / "models" / "drone_cnn_model.pth"
    check("drone_cnn_model.pth exists", wpath.exists(), str(wpath))
    model = cv_engine.get_cnn_model()
    check("weights loaded into PitWallCNN", cv_engine._CNN_TRAINED is True,
          "cnn_trained flag")

    print("\n=== 2. operating point ===")
    mpath = REPO / "models" / "cnn_metrics.json"
    check("cnn_metrics.json exists", mpath.exists())
    thr = cv_engine._decision_threshold()
    check("threshold within (0,1)", 0.0 < thr < 1.0, f"threshold={thr}")
    if mpath.exists():
        meta = json.loads(mpath.read_text())
        want = meta.get("test_at_tuned_threshold", {}).get("threshold")
        check("threshold comes from the metrics file, not a constant",
              want is not None and abs(float(want) - thr) < 1e-9,
              f"metrics={want} engine={thr}")

    print("\n=== 3. batched forward (the .view regression) ===")
    try:
        batch = torch.rand(4, 224, 224, 3).permute(0, 3, 1, 2)  # non-contiguous
        with torch.no_grad():
            out = model(batch)
        check("batch of 4 non-contiguous inputs", tuple(out.shape) == (4, 1),
              f"shape={tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001
        check("batch of 4 non-contiguous inputs", False, repr(exc))

    print("\n=== 4. response contract ===")
    data, src = sample_image_bytes()
    res = cv_engine.analyze_drone_pit_image(data)
    check("status success", res.get("status") == "success", f"image={src}")
    for key in ("anomaly_status", "mine_region_confidence_pct",
                "decision_threshold", "edge_gradient_intensity",
                "recommendation", "caveat", "cnn_trained", "cnn_architecture",
                "model_task"):
        check(f"key present: {key}", key in res)
    conf = res.get("mine_region_confidence_pct")
    check("confidence is a percentage",
          isinstance(conf, float) and 0.0 <= conf <= 100.0, f"{conf}")
    check("cnn_trained is True", res.get("cnn_trained") is True)

    print("\n=== 5. tiled scan ===")
    check("scanned more than one window on a full scene",
          res.get("tiles_scanned", 0) > 1, f"tiles={res.get('tiles_scanned')}")
    check("hit count never exceeds tiles scanned",
          res.get("tiles_above_threshold", 0) <= res.get("tiles_scanned", 0))
    check("coverage is a percentage",
          0.0 <= res.get("scene_coverage_pct", -1) <= 100.0,
          f"{res.get('scene_coverage_pct')}%")
    check("peak confidence >= mean confidence",
          res.get("mine_region_confidence_pct", 0)
          >= res.get("mean_tile_confidence_pct", 0))
    box = res.get("peak_tile_box")
    check("peak tile box is a 4-tuple", isinstance(box, list) and len(box) == 4,
          f"{box}")
    # A whole-scene image of a mine must not come back as background: that
    # was the train/serve mismatch the tiled scan exists to fix.
    if src != "<synthetic>":
        check("real pit scene classified as mine terrain",
              res.get("anomaly_status") == "OPEN-PIT MINE TERRAIN",
              f"{res.get('anomaly_status')} @ "
              f"{res.get('mine_region_confidence_pct')}%")

    print("\n=== 6. no crack claims ===")
    blob = json.dumps(res).lower()
    check("no 'crack_severity' key", "crack_severity" not in res)
    check("no 'visual_risk_percentage' key", "visual_risk_percentage" not in res)
    # The caveat is allowed to use the word 'crack' to deny it; nothing else is.
    stripped = json.dumps({k: v for k, v in res.items() if k != "caveat"}).lower()
    check("no crack claim outside the caveat", "crack" not in stripped)

    print("\n=== result ===")
    print(json.dumps(res, indent=2)[:700])
    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
