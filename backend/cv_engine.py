import io
import logging
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

class PitWallCNN(nn.Module):
    def __init__(self):
        super(PitWallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        # .reshape, not .view: a permuted (non-contiguous) input carries
        # its stride layout through the conv stack, and .view() raises
        # "view size is not compatible with input tensor's size and
        # stride" on the flatten. .reshape falls back to a copy when the
        # tensor cannot be viewed, and is identical otherwise.
        feat = feat.reshape(feat.size(0), -1)
        out = self.classifier(feat)
        return out

_CNN_MODEL = None
_CNN_TRAINED = False

def get_cnn_model():
    global _CNN_MODEL, _CNN_TRAINED
    if _CNN_MODEL is None:
        model = PitWallCNN()
        weights_path = os.path.join(MODELS_DIR, "drone_cnn_model.pth")
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=True)
                )
                _CNN_TRAINED = True
                logger.info("loaded trained CNN weights", extra={"weights_path": weights_path})
            except Exception as e:  # noqa: BLE001
                logger.warning("could not load CNN weights; running UNTRAINED", extra={"error": str(e)})
        else:
            logger.warning("no CNN weights found; CNN output is UNTRAINED and not meaningful")
        model.eval()
        _CNN_MODEL = model
    return _CNN_MODEL

_THRESHOLD = None


def _decision_threshold() -> float:
    """Operating point chosen on the validation split during training.

    ml/train_cnn.py tunes it for F1 and records it in cnn_metrics.json.
    At the default 0.5 this model is precision-heavy (0.98 precision /
    0.61 recall on held-out data); the tuned point trades a little
    precision for substantially better recall, which is the right
    direction for a safety-adjacent screen. Falls back to 0.5 when the
    metrics file is absent.
    """
    global _THRESHOLD
    if _THRESHOLD is None:
        _THRESHOLD = 0.5
        try:
            import json

            mp = os.path.join(MODELS_DIR, "cnn_metrics.json")
            with open(mp, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            thr = meta.get("test_at_tuned_threshold", {}).get("threshold")
            if isinstance(thr, (int, float)) and 0.0 < float(thr) < 1.0:
                _THRESHOLD = float(thr)
        except Exception:  # noqa: BLE001 -- absent or malformed: keep 0.5
            pass
    return _THRESHOLD


def analyze_drone_pit_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Classifies an aerial image patch as annotated open-pit mine terrain or
    background, using the CNN trained by ml/train_cnn.py.

    Held-out performance (models/cnn_metrics.json): ROC-AUC 0.956,
    precision 0.854, recall 0.909 at the tuned operating point, over a
    test split grouped by source scene.

    This is deliberately NOT a crack or slope-failure model. The only
    labels available for training were single-class bounding boxes around
    open-pit workings, so that is the only thing it can honestly report.
    """
    # Deferred import so the metrics dep doesn't leak into environments
    # that only run cv_engine standalone (e.g. the file's ``__main__``
    # smoke test).
    from app.core.metrics import observe_cv_inference

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((224, 224))
        img_np = np.array(image_resized, dtype=np.float32) / 255.0

        # Standard PyTorch normalization
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)

        # Run CNN Forward Pass -- timed for the Prometheus histogram.
        cnn_model = get_cnn_model()
        with observe_cv_inference(), torch.no_grad():
            raw_cnn_score = float(cnn_model(img_tensor).item())
            
        # Edge gradient magnitude. Reported as a plain texture statistic.
        # It is NOT blended into the model score: the previous version
        # averaged the CNN output with edge_intensity/50 and published the
        # result as a "visual risk percentage", which dressed an arbitrary
        # image-processing constant up as a model prediction.
        grayscale = image_resized.convert("L")
        arr = np.array(grayscale, dtype=np.float32)
        gy, gx = np.gradient(arr)
        edge_intensity = float(np.mean(np.hypot(gx, gy)))

        threshold = _decision_threshold()
        is_mine_terrain = raw_cnn_score >= threshold

        if is_mine_terrain:
            status = "OPEN-PIT MINE TERRAIN"
            recommendation = (
                "Scene matches annotated open-pit workings. Pair with sensor "
                "telemetry for any stability judgement -- this model says "
                "nothing about slope condition."
            )
        else:
            status = "BACKGROUND / NOT ANNOTATED MINE TERRAIN"
            recommendation = (
                "Scene does not match the annotated open-pit class. If you "
                "expected a pit here, check framing and altitude before "
                "reading anything into it."
            )

        return {
            "status": "success",
            "model_task": (
                "binary patch classification: annotated open-pit mine region "
                "vs background crop from the same aerial scene"
            ),
            "anomaly_status": status,
            "mine_region_confidence_pct": float(np.round(raw_cnn_score * 100.0, 1)),
            "decision_threshold": round(threshold, 3),
            "edge_gradient_intensity": float(np.round(edge_intensity, 2)),
            "recommendation": recommendation,
            "caveat": (
                "Trained to separate annotated open-pit mine regions from their "
                "surroundings. It does not detect cracks, does not estimate "
                "crack severity, and does not predict slope failure."
            ),
            "cnn_architecture": (
                "PyTorch PitWallCNN (trained on Open-Pit Mine Object Detection labels)"
                if _CNN_TRAINED
                else "PyTorch PitWallCNN (UNTRAINED -- output is not meaningful)"
            ),
            "cnn_trained": _CNN_TRAINED,
        }
        
    except Exception as e:  # noqa: BLE001
        logger.warning("drone image CNN analysis failed", extra={"error": str(e)})
        return {
            "status": "error",
            "message": f"Failed to analyze image: {str(e)}",
            "visual_risk_percentage": 0.0,
        }

if __name__ == "__main__":
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    dummy_img.save(img_byte_arr, format='JPEG')
    res = analyze_drone_pit_image(img_byte_arr.getvalue())
    print("CNN Analysis Test:\n", res)