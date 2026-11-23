"""Independent integrity checks for the frozen Jordan external-validation run."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "jordan_results"
RAW = ROOT / "raw"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import run_jordan_external_validation as runner
import revision_analysis as ra


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(left, right, name: str, tol: float = 1e-10):
    if not np.isclose(float(left), float(right), rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"{name}: {left!r} != {right!r}")


def main() -> None:
    checks = []
    lock_path = HERE / "transport_model_freeze.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, expected in lock["objects"].items():
        observed = sha256(HERE / "frozen_nhanes_models" / name)
        if observed != expected:
            raise AssertionError(f"Frozen object hash mismatch: {name}")
    checks.append({"check": "frozen_object_hashes", "status": "pass", "count": 4})

    manifest = json.loads((OUT / "jordan_external_validation_manifest.json").read_text(encoding="utf-8"))
    if sha256(RAW / manifest["source_file"]) != manifest["source_sha256"]:
        raise AssertionError("Jordan participant-file hash mismatch")
    if sha256(lock_path) != manifest["frozen_lock_sha256"]:
        raise AssertionError("Frozen lock hash mismatch")
    for flag in ["model_refit", "recalibration", "threshold_optimization", "synthetic_data_used"]:
        if manifest[flag] is not False:
            raise AssertionError(f"Unexpected manifest flag: {flag}")
    checks.append({"check": "source_and_lock_manifest", "status": "pass"})

    raw_dta = runner.read_data(RAW / "jor2019.dta")
    frame, flow, universe = runner.prepare(raw_dta)
    if len(frame) != 885 or int(frame.diabetes.sum()) != 249:
        raise AssertionError("Unexpected eligible cohort count")
    if universe.design_stratum.nunique() != 32 or universe.design_psu.nunique() != 750:
        raise AssertionError("Unexpected resampling universe")
    checks.append(
        {
            "check": "cohort_and_design",
            "status": "pass",
            "eligible_n": int(len(frame)),
            "cases": int(frame.diabetes.sum()),
            "domain_strata": int(frame.design_stratum.nunique()),
            "domain_psus": int(frame.design_psu.nunique()),
            "universe_psus": int(universe.design_psu.nunique()),
        }
    )

    # CSV and DTA exports must yield the same analysis IDs, outcomes, and derived predictors.
    raw_csv = runner.read_data(RAW / "jor2019.csv")
    frame_csv, _, universe_csv = runner.prepare(raw_csv)
    cols = [
        "age", "analysis_weight", "diabetes", "bmi", "waist_to_hip_ratio",
        "systolic_bp", "diastolic_bp", "heart_rate",
    ]
    left = frame.sort_values("id").reset_index(drop=True)
    right = frame_csv.sort_values("id").reset_index(drop=True)
    if not left.id.astype(str).equals(right.id.astype(str)):
        raise AssertionError("CSV and DTA eligible participant IDs differ")
    for col in cols:
        if not np.allclose(
            pd.to_numeric(left[col], errors="coerce"),
            pd.to_numeric(right[col], errors="coerce"),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ):
            raise AssertionError(f"CSV and DTA harmonization differs for {col}")
    if universe_csv.design_psu.nunique() != universe.design_psu.nunique():
        raise AssertionError("CSV and DTA design universes differ")
    checks.append({"check": "csv_dta_harmonization", "status": "pass"})

    results = pd.read_csv(OUT / "jordan_external_validation_results.csv").set_index("config")
    predictions = pd.read_csv(OUT / "jordan_external_validation_predictions.csv")
    if len(predictions) != len(frame) or predictions.diabetes.sum() != frame.diabetes.sum():
        raise AssertionError("Prediction row count or case count mismatch")
    metric_columns = [
        "pr_auc", "auc", "brier", "average_precision", "sensitivity", "specificity",
        "ppv", "npv", "accuracy", "f1", "calibration_intercept_joint",
        "calibration_slope", "calibration_in_the_large",
    ]
    for config in ["BASE", "BMI", "WHR", "BOTH"]:
        p = predictions[f"{config}_probability"]
        if p.isna().any() or not p.between(0, 1).all():
            raise AssertionError(f"Invalid probabilities for {config}")
        recalculated = ra.metrics(
            predictions.diabetes,
            p,
            predictions.analysis_weight,
            float(results.loc[config, "threshold"]),
        )
        for metric in metric_columns:
            assert_close(recalculated[metric], results.loc[config, metric], f"{config}_{metric}")
    checks.append({"check": "independent_metric_recalculation", "status": "pass", "values": 52})

    replicate_sets = {
        "overall": "jordan_external_validation",
        "Jordanian": "jordan_jordanian",
        "Syrian": "jordan_syrian",
    }
    interval_pairs_checked = 0
    for label, stem in replicate_sets.items():
        samples = pd.read_csv(OUT / f"{stem}_resampling_replicates.csv")
        intervals = pd.read_csv(OUT / f"{stem}_psu_intervals.csv", index_col="metric")
        if len(samples) != 2000:
            raise AssertionError(f"{label}: expected 2000 valid replicates")
        expected = samples.drop(columns="replicate").quantile([0.025, 0.975]).T
        for metric in expected.index:
            assert_close(expected.loc[metric, 0.025], intervals.loc[metric, "lower_psu"], f"{label}_{metric}_lower")
            assert_close(expected.loc[metric, 0.975], intervals.loc[metric, "upper_psu"], f"{label}_{metric}_upper")
            interval_pairs_checked += 1
    checks.append(
        {
            "check": "resampling_endpoints",
            "status": "pass",
            "replicate_sets": 3,
            "replicates_each": 2000,
            "interval_pairs": interval_pairs_checked,
        }
    )

    report = {
        "status": "pass",
        "verified_at_local_date": "2026-10-27",
        "checks": checks,
    }
    (OUT / "jordan_external_validation_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
