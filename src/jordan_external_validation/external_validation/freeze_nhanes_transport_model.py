"""Freeze reduced NHANES models before participant-level Jordan data are accessed.

The script fits only prespecified logistic models in the existing PSU-disjoint
partitions. Preprocessing is fitted in training, calibration in validation, and
the threshold is fixed at 0.20. A lock file prevents accidental refitting.

Package version: shared NHANES code (design.py, revision_analysis.py) and the frozen PSU assignment are read from
../../nhanes, which are byte-identical to the files used when the models were frozen on 20 October 2026.
The NHANES pipeline must be run first (nhanes/download_sources.py and nhanes/run_all.py) so the prepared data exist.

Reproduce without touching the submitted frozen models:
    python freeze_nhanes_transport_model.py --output my_refit
The submitted frozen models and lock are only overwritten with --force (not needed for checking).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
NHANES_CODE = HERE.parents[1] / "nhanes"
DEST = HERE / "frozen_nhanes_models"
LOCK = HERE / "transport_model_freeze.lock.json"
SEED = 42
THRESHOLD = 0.20

CORE_NUMERIC = ["age", "systolic_bp", "diastolic_bp", "heart_rate"]
CATEGORICAL = ["hypertension_history"]
CONFIGS = {
    "BASE": [],
    "BMI": ["bmi"],
    "WHR": ["waist_to_hip_ratio"],
    "BOTH": ["bmi", "waist_to_hip_ratio"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


def preprocessor(numeric: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("impute", KNNImputer(n_neighbors=5, keep_empty_features=True)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "encode",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, CATEGORICAL)]
    )


def main(force: bool = False, output: str | None = None) -> None:
    global DEST, LOCK
    if output:  # refit into a separate folder for comparison; the submitted frozen models stay untouched
        DEST = Path(output).resolve()
        LOCK = DEST / "transport_model_freeze.lock.json"
    if LOCK.exists() and not force:
        raise RuntimeError(
            f"Frozen-model lock already exists: {LOCK}. Refusing to refit. "
            "Use --force only if the prespecified protocol is deliberately amended."
        )

    import sys

    sys.path.insert(0, str(NHANES_CODE))
    import design
    import revision_analysis as ra

    d = design.load("mec_nonfasting")
    d = d[(d.age >= 18) & (d.age <= 69)].copy()
    if not set(d.DIQ010.unique()).issubset({1, 2}):
        raise AssertionError("Only definite DIQ010 responses may enter the frozen model.")
    if (d.analysis_weight <= 0).any():
        raise AssertionError("All included NHANES analysis weights must be positive.")

    split_sets = {s: set(d.loc[d.split == s, "psu_group"]) for s in ["train", "validation", "test"]}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not split_sets[left].isdisjoint(split_sets[right]):
            raise AssertionError(f"PSU leakage between {left} and {right}.")

    DEST.mkdir(parents=True, exist_ok=True)
    tr = d[d.split == "train"].copy()
    va = d[d.split == "validation"].copy()
    te = d[d.split == "test"].copy()

    cohort_rows = []
    for name, frame in [("all", d), ("train", tr), ("validation", va), ("test", te)]:
        cohort_rows.append(
            {
                "split": name,
                "n": len(frame),
                "cases": int(frame.diabetes.sum()),
                "noncases": int((frame.diabetes == 0).sum()),
                "psus": int(frame.psu_group.nunique()),
                "strata": int(frame.SDMVSTRA.nunique()),
                "weight_total": float(frame.analysis_weight.sum()),
                "weighted_case_fraction": float(
                    np.average(frame.diabetes, weights=frame.analysis_weight)
                ),
            }
        )
    pd.DataFrame(cohort_rows).to_csv(DEST / "nhanes_cohort_and_split.csv", index=False)

    missing_cols = CORE_NUMERIC + ["bmi", "waist_to_hip_ratio"] + CATEGORICAL
    missing = []
    for split_name, frame in [("train", tr), ("validation", va), ("test", te)]:
        for col in missing_cols:
            missing.append(
                {
                    "split": split_name,
                    "variable": col,
                    "missing_n": int(frame[col].isna().sum()),
                    "missing_percent": float(frame[col].isna().mean() * 100),
                }
            )
    pd.DataFrame(missing).to_csv(DEST / "nhanes_missingness.csv", index=False)

    rows = []
    prediction_frame = te[
        ["SEQN", "diabetes", "analysis_weight", "SDMVSTRA", "SDMVPSU", "psu_group"]
    ].copy()
    object_hashes = {}
    for config, anthropometry in CONFIGS.items():
        numeric = CORE_NUMERIC + anthropometry
        features = numeric + CATEGORICAL
        prep = preprocessor(numeric)
        z_train = prep.fit_transform(tr[features])
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )
        model.fit(
            z_train,
            tr.diabetes,
            sample_weight=tr.analysis_weight.to_numpy() / tr.analysis_weight.mean(),
        )

        validation_score = model.decision_function(prep.transform(va[features]))
        calibration = ra.calibrate(
            validation_score,
            va.diabetes,
            va.analysis_weight,
            "logit",
        )
        test_score = model.decision_function(prep.transform(te[features]))
        probability = ra.apply_calibrator(calibration, test_score, "logit")
        prediction_frame[f"{config}_probability"] = probability

        result = {
            "config": config,
            "model": "survey-weighted class-balanced logistic regression",
            "features": features,
            "train_n": len(tr),
            "validation_n": len(va),
            "test_n": len(te),
            "test_cases": int(te.diabetes.sum()),
            "threshold": THRESHOLD,
            **ra.metrics(te.diabetes, probability, te.analysis_weight, THRESHOLD),
        }
        ci, note = ra.bootstrap_metrics(te, probability, replicates=500)
        result["uncertainty_note"] = note
        result["validation_calibration_intercept"] = float(calibration.intercept_[0])
        result["validation_calibration_slope"] = float(calibration.coef_[0, 0])
        ci.to_csv(DEST / f"{config}_internal_test_psu_intervals.csv", index_label="metric")

        fitted = {
            "preprocessor": prep,
            "model": model,
            "calibrator": calibration,
            "score_kind": "logit",
            "features": features,
            "numeric_features": numeric,
            "categorical_features": CATEGORICAL,
            "threshold": THRESHOLD,
            "development_population": "NHANES 2017-2018 men aged 18-69 with definite DIQ010 and positive WTMEC2YR",
            "development_outcome": "cross-sectional self-reported diagnosed diabetes; type unspecified",
            "external_outcome": "reported prior diagnosis of raised blood sugar or diabetes; related but nonidentical",
            "partition": "existing disjoint masked-PSU-pair partition",
            "external_refitting_permitted": False,
            "external_recalibration_permitted": False,
        }
        path = DEST / f"{config}_LogReg_frozen.joblib"
        joblib.dump(fitted, path)
        object_hashes[path.name] = sha256(path)
        rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(DEST / "nhanes_internal_test_results.csv", index=False)
    prediction_frame.to_csv(DEST / "nhanes_internal_test_predictions.csv", index=False)

    comparison = {
        "BMI_minus_WHR_auc": float(
            results.loc[results.config == "BMI", "auc"].iloc[0]
            - results.loc[results.config == "WHR", "auc"].iloc[0]
        ),
        "interpretation": "Internal held-out comparison for the reduced transportable models; external conclusion remains pending Jordan participant data.",
    }
    dump(DEST / "nhanes_bmi_vs_whr_comparison.json", comparison)

    protocol = HERE / "HARMONIZATION_PROTOCOL.md"
    crosswalk = HERE / "variable_crosswalk.csv"
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "seed": SEED,
        "threshold": THRESHOLD,
        "model_family": "logistic regression only",
        "calibration": "weighted logistic calibration fitted in NHANES validation partition",
        "external_refitting_permitted": False,
        "external_recalibration_permitted": False,
        "participant_level_jordan_data_accessed_before_freeze": False,
        "synthetic_data_used": False,
        "protocol_sha256": sha256(protocol),
        "crosswalk_sha256": sha256(crosswalk),
        "nhanes_psu_assignment_sha256": sha256(NHANES_CODE / "protocol" / "psu_assignment.csv"),
        "objects": object_hashes,
    }
    dump(LOCK, manifest)
    dump(DEST / "freeze_manifest.json", manifest)
    print(pd.DataFrame(cohort_rows).to_string(index=False))
    print(results[["config", "auc", "brier", "calibration_slope", "calibration_in_the_large"]].to_string(index=False))
    print(f"Frozen objects written to: {DEST}")
    print(f"Lock written to: {LOCK}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", help="write a refit to this folder instead of the submitted frozen models")
    args = parser.parse_args()
    main(force=args.force, output=args.output)

