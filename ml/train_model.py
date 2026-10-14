"""RockfallGuard risk model training pipeline.

=============================================================================
READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS
=============================================================================
This script trains on SYNTHETIC data generated from a documented physical
model (see `generate_dataset`). It does NOT train on real instrumented
open-pit slope measurements, because no such public dataset exists.

Therefore:

  * Metrics produced here measure how well the model recovers the generating
    function. They are NOT a claim about real-world rockfall prediction.
  * A high score here is expected and largely uninformative. If a model
    cannot recover a function you yourself wrote down, something is broken.
  * Any figure from this script must be reported as "synthetic benchmark",
    never as "accuracy" without qualification.

Two methodological choices that matter, and why:

1. TEMPORAL SPLIT, NOT RANDOM SPLIT. Sensor readings are a time series;
   consecutive frames are near-identical. A random split puts frame N in
   train and frame N+1 in test, so the model is tested on data it has
   effectively seen. That single mistake is the most common source of
   inflated accuracy in time-series ML, and it inflates scores far more
   than any modelling choice. Splits here are by contiguous time block.

2. ACCURACY IS THE WRONG HEADLINE METRIC. In a realistic deployment the
   Critical class is rare. A model that predicts "Safe" always scores well
   on accuracy and kills people. This script reports per-class precision
   and recall, PR-AUC for the Critical class, and a confusion matrix; the
   number that actually matters is Critical-class RECALL (what fraction of
   genuine failures we caught).

Usage:
    python ml/train_model.py                 # train + evaluate + save
    python ml/train_model.py --no-save       # evaluate only
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
MODELS_DIR: Final[Path] = REPO_ROOT / "models"

# Must match ml_engine.py exactly. A mismatch here is silent and catastrophic:
# the scaler would apply the wrong column's statistics at inference time.
FEATURE_NAMES: Final[list[str]] = [
    "rainfall_mm",
    "humidity_pct",
    "pore_pressure_kpa",
    "displacement_mm",
    "velocity_mm_h",
    "acceleration_mm_h2",
    "filtered_seismic_g",
    "hydro_kinematic_index",
    "fracture_instability_index",
    "rain_rolling_6h",
    "vel_rolling_mean",
    "vel_rolling_max",
    "pore_rolling_mean",
]

CLASS_LABELS: Final[list[str]] = ["Safe", "Warning", "Critical"]
RANDOM_SEED: Final[int] = 42


@dataclass
class TrainingConfig:
    n_sites: int = 40
    hours_per_site: int = 720  # 30 days of hourly readings per site
    # Deliberately imbalanced to reflect reality: genuine pre-failure states
    # are rare. Training on a balanced set and reporting accuracy is exactly
    # how you get a number that looks great and means nothing.
    critical_fraction: float = 0.04
    warning_fraction: float = 0.16
    noise_level: float = 0.25
    test_fraction: float = 0.2
    val_fraction: float = 0.15
    seed: int = RANDOM_SEED


@dataclass
class EvaluationResult:
    """Metrics bundle. Serialised to models/metrics.json for CI to read."""

    critical_recall: float
    critical_precision: float
    critical_pr_auc: float
    critical_threshold: float
    macro_f1: float
    accuracy: float
    regressor_mae: float
    regressor_r2: float
    confusion: list[list[int]]
    per_class: dict = field(default_factory=dict)
    data_provenance: str = "synthetic"
    n_train: int = 0
    n_test: int = 0


def _velocity_regime(rng: np.random.Generator, n: int, regime: str) -> np.ndarray:
    """Creep velocity by stability regime, in mm/h.

    Ranges follow the inverse-velocity literature on progressive slope failure:
    stable creep is sub-mm/h, tertiary (pre-failure) creep accelerates sharply.

    The distributions deliberately OVERLAP. An earlier version of this
    generator used well-separated distributions and the classifier scored
    1.000 precision and recall on the Critical class — which tells you nothing
    except that the classes were separable by construction. Real slopes do not
    announce themselves that cleanly: some fail with little precursory
    acceleration, and haul-truck activity produces readings that look alarming
    and are not. If your synthetic benchmark returns a perfect score, the
    benchmark is broken, not the model.
    """
    if regime == "critical":
        return rng.lognormal(mean=-0.5, sigma=1.0, size=n)
    if regime == "warning":
        return rng.lognormal(mean=-1.3, sigma=0.9, size=n)
    return rng.lognormal(mean=-2.4, sigma=0.9, size=n)


def generate_dataset(config: TrainingConfig) -> pd.DataFrame:
    """Generate a synthetic multi-site sensor time series.

    Each site gets a contiguous hourly series so the temporal split below is
    meaningful. Site geometry (slope angle, pit depth) varies to stop the model
    learning a single global threshold.
    """
    rng = np.random.default_rng(config.seed)
    frames: list[pd.DataFrame] = []

    for site_id in range(config.n_sites):
        n = config.hours_per_site
        slope_angle = rng.uniform(30.0, 65.0)
        pit_depth = rng.uniform(100.0, 800.0)
        geometry_stress = (slope_angle / 45.0) * (1.0 + (pit_depth - 150.0) / 2000.0)

        # Assign a regime per hour, with persistence: a slope does not flip
        # between safe and critical hour to hour. Independent sampling would
        # make the problem artificially easy and destroy the time structure.
        regimes = np.empty(n, dtype=object)
        current = "safe"
        for t in range(n):
            roll = rng.random()
            if current == "safe":
                current = "warning" if roll < 0.015 else "safe"
            elif current == "warning":
                if roll < 0.04:
                    current = "critical"
                elif roll < 0.20:
                    current = "safe"
            else:  # critical
                if roll < 0.12:
                    current = "warning"
            regimes[t] = current

        # Rainfall: seasonal signal + stochastic storms.
        hours = np.arange(n)
        seasonal = 2.0 * (1.0 + np.sin(2 * np.pi * hours / (24 * 14)))
        storms = rng.gamma(shape=0.4, scale=6.0, size=n) * (rng.random(n) < 0.12)
        rainfall = np.clip(seasonal * rng.random(n) + storms, 0.0, None)
        # Antecedent rainfall drives pore pressure with a lag — this is the
        # physically real mechanism, and the reason rain_rolling_6h exists.
        rain_6h = pd.Series(rainfall).rolling(6, min_periods=1).sum().to_numpy()

        velocity = np.concatenate(
            [_velocity_regime(rng, 1, str(r)) for r in regimes]
        ) * geometry_stress
        velocity = np.clip(velocity + rng.normal(0, config.noise_level * 0.05, n), 0.001, None)

        # Silent failures. A real fraction of slope collapses give little or no
        # precursory acceleration — brittle, structurally-controlled failures in
        # particular. Modelling them matters: without them the dataset implies
        # every failure is predictable, which puts a recall ceiling of 1.0 within
        # reach and teaches you nothing about the irreducible miss rate. Applied
        # BEFORE the rolling/derivative features so those inherit the suppression.
        silent = (regimes == "critical") & (rng.random(n) < 0.18)
        velocity = np.where(silent, velocity * rng.uniform(0.05, 0.15, n), velocity)

        vel_series = pd.Series(velocity)
        vel_rolling_mean = vel_series.rolling(6, min_periods=1).mean().to_numpy()
        vel_rolling_max = vel_series.rolling(6, min_periods=1).max().to_numpy()

        acceleration = np.gradient(velocity)
        displacement = np.cumsum(velocity) + rng.uniform(5.0, 30.0)

        pore_pressure = (
            35.0 * geometry_stress
            + 1.8 * rain_6h
            + np.where(regimes == "critical", rng.uniform(0, 18, n), 0.0)
            + rng.normal(0, 6.0 * config.noise_level, n)
        ).clip(5.0, None)
        pore_rolling_mean = pd.Series(pore_pressure).rolling(6, min_periods=1).mean().to_numpy()

        humidity = np.clip(45.0 + 1.5 * rain_6h + rng.normal(0, 8, n), 10.0, 100.0)

        # Machinery noise is a confounder by design: haul trucks produce high
        # seismic RMS with no instability. If the model keys on raw seismic it
        # will false-positive on every truck, which is the single most likely
        # real-world failure mode for this system.
        machinery = (rng.random(n) < 0.15) & (regimes == "safe")

        # Microseismic emission by regime, with OVERLAPPING distributions.
        # An earlier version gave elevated seismic to Critical only and left
        # Warning at the Safe baseline, which made filtered_seismic_g a perfect
        # single-feature discriminator and produced a fake 1.000 Critical score.
        # Watch for this: a lone feature that separates a class cleanly is
        # almost always a leak in how the data was constructed.
        seismic_by_regime = np.select(
            [regimes == "critical", regimes == "warning"],
            [np.abs(rng.normal(0.15, 0.10, n)), np.abs(rng.normal(0.08, 0.06, n))],
            default=np.abs(rng.normal(0.025, 0.02, n)),
        )
        raw_seismic = np.where(machinery, np.abs(rng.normal(0.32, 0.09, n)), seismic_by_regime)
        # Silent failures are seismically quiet too, not just slow.
        raw_seismic = np.where(silent, raw_seismic * rng.uniform(0.1, 0.25, n), raw_seismic)
        # The low-pass filter is what separates genuine microseismic emission
        # from machinery: it attenuates the machinery band far more.
        filtered_seismic = np.where(machinery, raw_seismic * 0.12, raw_seismic * 0.75)

        frames.append(
            pd.DataFrame(
                {
                    "site_id": site_id,
                    "t": hours,
                    "rainfall_mm": rainfall,
                    "humidity_pct": humidity,
                    "pore_pressure_kpa": pore_pressure,
                    "displacement_mm": displacement,
                    "velocity_mm_h": velocity,
                    "acceleration_mm_h2": acceleration,
                    "filtered_seismic_g": filtered_seismic,
                    "raw_seismic_rms_g": raw_seismic,
                    "hydro_kinematic_index": pore_pressure * velocity,
                    "fracture_instability_index": acceleration * filtered_seismic,
                    "rain_rolling_6h": rain_6h,
                    "vel_rolling_mean": vel_rolling_mean,
                    "vel_rolling_max": vel_rolling_max,
                    "pore_rolling_mean": pore_rolling_mean,
                    "regime": regimes,
                    "is_machinery": machinery,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["regime"].map({"safe": 0, "warning": 1, "critical": 2}).astype(int)

    # Continuous risk target, monotonic in the drivers, with noise so the
    # regressor cannot invert it exactly.
    df["risk_percentage"] = np.clip(
        12.0 * np.log1p(df["velocity_mm_h"] * 40)
        + 0.22 * df["pore_pressure_kpa"]
        + 25.0 * df["filtered_seismic_g"]
        + np.random.default_rng(config.seed).normal(0, 4.0, len(df)),
        0.0,
        100.0,
    )
    return df


def temporal_split(
    df: pd.DataFrame, config: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each site's series by time: earliest → train, latest → test.

    This is the whole point. A random split leaks: adjacent hourly readings are
    nearly identical, so the test set becomes a near-duplicate of the training
    set and every metric inflates. Splitting forward in time reproduces the real
    deployment condition — predict the future from the past.
    """
    train, val, test = [], [], []
    for _, group in df.groupby("site_id", sort=False):
        group = group.sort_values("t")
        n = len(group)
        test_start = int(n * (1 - config.test_fraction))
        val_start = int(test_start * (1 - config.val_fraction))
        train.append(group.iloc[:val_start])
        val.append(group.iloc[val_start:test_start])
        test.append(group.iloc[test_start:])
    return (
        pd.concat(train, ignore_index=True),
        pd.concat(val, ignore_index=True),
        pd.concat(test, ignore_index=True),
    )


def tune_critical_threshold(
    classifier, scaler, val_df: pd.DataFrame, target_recall: float = 0.85
) -> float:
    """Pick the Critical-class probability threshold on the VALIDATION set.

    `argmax` over class probabilities implicitly assumes every error costs the
    same. It does not. Missing a slope failure can kill someone; a false alarm
    costs an unnecessary evacuation and some credibility. So the threshold is
    tuned for recall and the precision cost is accepted and reported.

    Tuned on validation, never on test — tuning on test is how you end up
    reporting a number you cannot reproduce in deployment.
    """
    x_val = scaler.transform(val_df[FEATURE_NAMES])
    proba_critical = classifier.predict_proba(x_val)[:, 2]
    y_critical = (val_df["label"].to_numpy() == 2).astype(int)

    if y_critical.sum() == 0:
        return 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = (proba_critical >= threshold).astype(int)
        recall = predicted[y_critical == 1].mean() if y_critical.sum() else 0.0
        if recall < target_recall:
            return float(max(threshold - 0.01, 0.01))
    return 0.5


def evaluate(
    classifier,
    regressor,
    scaler,
    test_df: pd.DataFrame,
    n_train: int,
    critical_threshold: float = 0.5,
) -> EvaluationResult:
    """Compute the metrics that actually matter for a safety system."""
    x_test = scaler.transform(test_df[FEATURE_NAMES])
    y_true = test_df["label"].to_numpy()
    y_proba = classifier.predict_proba(x_test)

    # Apply the recall-tuned threshold: anything above it is escalated to
    # Critical regardless of which class argmax would have chosen.
    y_pred = classifier.predict(x_test)
    y_pred = np.where(y_proba[:, 2] >= critical_threshold, 2, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_LABELS, output_dict=True, zero_division=0
    )
    critical = report["Critical"]

    # PR-AUC, not ROC-AUC: with a rare positive class, ROC-AUC looks good even
    # for a model with poor precision, because true negatives dominate.
    critical_pr_auc = float(
        average_precision_score((y_true == 2).astype(int), y_proba[:, 2])
    )

    risk_pred = regressor.predict(x_test)
    return EvaluationResult(
        critical_recall=float(critical["recall"]),
        critical_precision=float(critical["precision"]),
        critical_pr_auc=critical_pr_auc,
        critical_threshold=float(critical_threshold),
        macro_f1=float(report["macro avg"]["f1-score"]),
        accuracy=float(report["accuracy"]),
        regressor_mae=float(mean_absolute_error(test_df["risk_percentage"], risk_pred)),
        regressor_r2=float(r2_score(test_df["risk_percentage"], risk_pred)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
        per_class={
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASS_LABELS
        },
        n_train=n_train,
        n_test=len(test_df),
    )


def print_report(result: EvaluationResult, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("  EVALUATION — SYNTHETIC DATA. NOT A REAL-WORLD PERFORMANCE CLAIM.")
    print("=" * 74)
    print(f"  Train / test rows: {result.n_train} / {result.n_test} (temporal split)")
    print(f"  Class balance in test: {dict(test_df['label'].value_counts().sort_index())}")
    print()
    print("  Per class:")
    print(f"    {'':10s} {'precision':>10s} {'recall':>10s} {'f1':>8s} {'support':>9s}")
    for label, m in result.per_class.items():
        print(
            f"    {label:10s} {m['precision']:10.3f} {m['recall']:10.3f} "
            f"{m['f1']:8.3f} {m['support']:9d}"
        )
    print()
    print("  Headline metrics (Critical class is the one that matters):")
    print(f"    Critical recall     {result.critical_recall:.3f}   <- fraction of real failures caught")
    print(f"    Critical precision  {result.critical_precision:.3f}   <- fraction of alarms that were real")
    print(f"    Critical PR-AUC     {result.critical_pr_auc:.3f}")
    print(f"    Decision threshold  {result.critical_threshold:.2f}   <- tuned on validation for recall")
    print(f"    Macro F1            {result.macro_f1:.3f}")
    print(f"    Accuracy            {result.accuracy:.3f}   <- do NOT quote this alone")
    print()
    print(f"  Risk regressor: MAE {result.regressor_mae:.2f} pct-points, R2 {result.regressor_r2:.3f}")
    print()
    print("  Confusion matrix (rows = actual, cols = predicted):")
    print(f"    {'':10s}" + "".join(f"{c:>10s}" for c in CLASS_LABELS))
    for label, row in zip(CLASS_LABELS, result.confusion):
        print(f"    {label:10s}" + "".join(f"{v:10d}" for v in row))
    print()
    missed = result.confusion[2][0] + result.confusion[2][1]
    print(f"  MISSED CRITICAL EVENTS: {missed} of {sum(result.confusion[2])}")
    print("  In deployment each of these is an unwarned slope failure.")
    print("=" * 74 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="evaluate without writing artifacts")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    config = TrainingConfig(seed=args.seed)
    logger.info("Generating synthetic dataset (%d sites)...", config.n_sites)
    df = generate_dataset(config)
    logger.info("Generated %d rows. Class balance: %s", len(df), dict(df["label"].value_counts().sort_index()))

    train_df, val_df, test_df = temporal_split(df, config)
    logger.info(
        "Temporal split — train %d / val %d / test %d", len(train_df), len(val_df), len(test_df)
    )

    scaler = StandardScaler().fit(train_df[FEATURE_NAMES])
    x_train = scaler.transform(train_df[FEATURE_NAMES])

    logger.info("Training classifier...")
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        # Without this, the rare Critical class is effectively ignored — the
        # model minimises loss by predicting Safe.
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=-1,
    ).fit(x_train, train_df["label"])

    logger.info("Training risk regressor...")
    regressor = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=config.seed
    ).fit(x_train, train_df["risk_percentage"])

    threshold = tune_critical_threshold(classifier, scaler, val_df, target_recall=0.85)
    logger.info("Critical threshold tuned on validation: %.2f (target recall 0.85)", threshold)
    result = evaluate(
        classifier, regressor, scaler, test_df, n_train=len(train_df),
        critical_threshold=threshold,
    )
    print_report(result, test_df)

    if args.no_save:
        logger.info("--no-save set; artifacts not written.")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS_DIR / "rockfall_classifier.joblib")
    joblib.dump(regressor, MODELS_DIR / "rockfall_regressor.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(FEATURE_NAMES, MODELS_DIR / "feature_names.joblib")

    # SHAP is optional: the API degrades to no explanations rather than failing.
    try:
        import shap

        explainer = shap.TreeExplainer(classifier)
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
        logger.info("SHAP explainer saved.")
    except ImportError:
        logger.warning("shap not installed — skipping explainer (API will omit explanations)")

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "synthetic — generated by ml/train_model.py, not real sensor data",
        "split_strategy": "temporal (per-site, forward in time)",
        "warning": "These metrics measure recovery of a known generating function. "
        "They are NOT a real-world rockfall prediction accuracy claim.",
        **{k: v for k, v in vars(result).items()},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Artifacts written to %s", MODELS_DIR)

    # Guardrail, not a marketing target. Catches a genuinely broken pipeline
    # (wrong column order, all-one-class predictions) without asserting an
    # impressive-sounding number.
    # A deliberately LOW floor. Its job is to catch a structurally broken
    # pipeline -- shuffled columns, a scaler fit on the wrong split, a model
    # collapsing to a single class -- not to enforce an impressive number.
    # Setting this high would recreate exactly the incentive that produced the
    # original "99.83%" claim: a threshold you tune the data to clear.
    if result.critical_recall < 0.30 or result.critical_pr_auc < 0.30:
        logger.error(
            "Critical recall %.3f / PR-AUC %.3f below the sanity floor (0.30). "
            "This indicates a broken pipeline, not merely a weak model.",
            result.critical_recall, result.critical_pr_auc,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())