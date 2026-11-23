"""Apply the frozen NHANES transport models unchanged to Jordan STEPS 2019.

Participant data must be obtained through the WHO microdata request. This script
does not download data and refuses to run if the frozen-object hashes differ from
the pre-access lock file.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = HERE / "frozen_nhanes_models"
LOCK = HERE / "transport_model_freeze.lock.json"
CONFIGS = ["BASE", "BMI", "WHR", "BOTH"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix == ".dta":
        frame = pd.read_stata(path, convert_categoricals=False)
    elif suffix == ".sav":
        frame = pd.read_spss(path, convert_categoricals=False)
    elif suffix in {".sas7bdat", ".xpt"}:
        frame = pd.read_sas(path, format="xport" if suffix == ".xpt" else "sas7bdat")
    else:
        raise ValueError(f"Unsupported participant-file extension: {suffix}")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after lower-case normalization.")
    return frame


def numeric(frame: pd.DataFrame, name: str, sentinels=()) -> pd.Series:
    x = pd.to_numeric(frame[name], errors="coerce")
    if sentinels:
        x = x.mask(x.isin(sentinels))
    return x


def mean_readings(frame: pd.DataFrame, names: list[str], low: float, high: float) -> pd.Series:
    values = pd.concat(
        [numeric(frame, name, sentinels=(666, 777, 888, 999)) for name in names], axis=1
    )
    values = values.where(values.ge(low) & values.le(high))
    return values.mean(axis=1).where(values.notna().sum(axis=1) >= 2)


def sex_is_male(series: pd.Series) -> pd.Series:
    number = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    # The Jordan public file uses the plural label "Men" in both CSV and DTA
    # exports. Accept it as the same prespecified male category.
    return number.eq(1) | text.isin({"male", "man", "men", "m"})


def prepare(raw: pd.DataFrame):
    required = {
        "id", "nationality", "psu", "stratum", "wstep2", "sex", "age", "h6", "h7a", "h2a",
        "m4a", "m5a", "m6a", "m4b", "m5b", "m6b", "m16a", "m16b", "m16c",
        "m11", "m12", "m14", "m15",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Jordan file is missing required variables: {missing}")
    if raw["id"].duplicated().any():
        raise ValueError("Participant ID is not unique.")

    age = numeric(raw, "age", sentinels=(77, 88, 99, 777, 888, 999))
    h6 = numeric(raw, "h6", sentinels=(77, 88, 99))
    h7a = numeric(raw, "h7a", sentinels=(77, 88, 99))
    weight = numeric(raw, "wstep2")
    base_mask = sex_is_male(raw["sex"]) & age.between(18, 69) & weight.gt(0)
    outcome_mask = h6.eq(1) & h7a.isin([1, 2])
    eligible = base_mask & outcome_mask

    d = pd.DataFrame(index=raw.index)
    d["id"] = raw["id"]
    d["nationality"] = raw["nationality"]
    d["age"] = age
    d["analysis_weight"] = weight
    d["diabetes"] = h7a.eq(1).astype(int)
    height = numeric(raw, "m11", sentinels=(666, 777, 888, 999)).where(lambda x: x.between(100, 270))
    body_weight = numeric(raw, "m12", sentinels=(666, 777, 888, 999)).where(lambda x: x.between(20, 350))
    waist = numeric(raw, "m14", sentinels=(666, 777, 888, 999)).where(lambda x: x.between(30, 200))
    hip = numeric(raw, "m15", sentinels=(666, 777, 888, 999)).where(lambda x: x.between(45, 300))
    d["bmi"] = body_weight / (height / 100.0) ** 2
    d["waist_to_hip_ratio"] = waist / hip
    d["systolic_bp"] = mean_readings(raw, ["m4a", "m5a", "m6a"], 40, 300)
    d["diastolic_bp"] = mean_readings(raw, ["m4b", "m5b", "m6b"], 30, 200)
    d["heart_rate"] = mean_readings(raw, ["m16a", "m16b", "m16c"], 30, 200)
    h2a = numeric(raw, "h2a", sentinels=(77, 88, 99))
    d["hypertension_history"] = h2a.map({1: "Yes", 2: "No"})
    # Jordanians and Syrians were sampled separately. Including nationality in
    # public design identifiers prevents accidental merging of separate designs.
    d["design_stratum"] = raw["nationality"].astype(str) + "|" + raw["stratum"].astype(str)
    d["design_psu"] = raw["nationality"].astype(str) + "|" + raw["psu"].astype(str)

    flow = [
        {"stage": "Public-use file", "n": len(raw)},
        {"stage": "Men aged 18-69", "n": int((sex_is_male(raw.sex) & age.between(18, 69)).sum())},
        {"stage": "Men aged 18-69 with positive wstep2", "n": int(base_mask.sum())},
        {"stage": "Previously had blood sugar measured (h6=1)", "n": int((base_mask & h6.eq(1)).sum())},
        {"stage": "Primary external outcome observed", "n": int(eligible.sum())},
    ]
    # Retain the full positive-Step-2 PSU universe for domain resampling.  The
    # external outcome is observed only in the h6=1 domain, and one stratum has
    # only one PSU represented in that domain even though both sampled PSUs are
    # present in the public file.  Sampling from the full design universe and
    # calculating metrics in the eligible domain preserves that sampled PSU.
    design_universe_mask = weight.gt(0)
    design_universe = pd.DataFrame(index=raw.index[design_universe_mask])
    design_universe["nationality"] = raw.loc[design_universe_mask, "nationality"].astype(str)
    design_universe["design_stratum"] = (
        raw.loc[design_universe_mask, "nationality"].astype(str)
        + "|"
        + raw.loc[design_universe_mask, "stratum"].astype(str)
    )
    design_universe["design_psu"] = (
        raw.loc[design_universe_mask, "nationality"].astype(str)
        + "|"
        + raw.loc[design_universe_mask, "psu"].astype(str)
    )
    design_universe = design_universe.drop_duplicates().reset_index(drop=True)
    return d.loc[eligible].copy(), pd.DataFrame(flow), design_universe


def verify_frozen_objects() -> dict:
    if not LOCK.exists():
        raise FileNotFoundError("Frozen-model lock is absent. Run freeze_nhanes_transport_model.py first.")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("participant_level_jordan_data_accessed_before_freeze") is not False:
        raise RuntimeError("Lock does not attest that freezing preceded Jordan participant-data access.")
    for name, expected in lock["objects"].items():
        path = MODEL_DIR / name
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen object hash mismatch: {name}")
    return lock


def metric_function(y, p, w, threshold):
    import sys
    sys.path.insert(0, str(ROOT))
    import revision_analysis as ra
    return ra.metrics(y, p, w, threshold)


def resample_intervals(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    threshold: float,
    replicates: int,
    design_universe: pd.DataFrame,
):
    rng = np.random.default_rng(42)
    strata_units = []
    for stratum, block in design_universe.groupby("design_stratum"):
        units = block.design_psu.unique()
        if len(units) < 2:
            raise RuntimeError(f"Design stratum {stratum!r} contains fewer than two sampled PSUs.")
        strata_units.append(units)
    groups = frame.design_psu.to_numpy()
    base_weight = frame.analysis_weight.to_numpy()
    y = frame.diabetes.to_numpy()
    records = []
    for replicate in range(replicates):
        multiplier = np.zeros(len(frame))
        for units in strata_units:
            m = len(units)
            selected = rng.choice(units, m - 1, replace=True)
            for unit in units:
                multiplier[groups == unit] = m / (m - 1) * np.sum(selected == unit)
        w = base_weight * multiplier
        if np.unique(y[w > 0]).size < 2:
            continue
        record = {"replicate": replicate}
        for config, p in probabilities.items():
            met = metric_function(y, p, w, threshold)
            for key, value in met.items():
                if key != "calibration_fit_converged":
                    record[f"{config}_{key}"] = value
        record["BMI_minus_WHR_auc"] = record["BMI_auc"] - record["WHR_auc"]
        record["WHR_minus_BMI_auc"] = record["WHR_auc"] - record["BMI_auc"]
        record["BOTH_minus_BMI_auc"] = record["BOTH_auc"] - record["BMI_auc"]
        records.append(record)
    samples = pd.DataFrame(records)
    if len(samples) < max(100, replicates // 2):
        raise RuntimeError("Too few valid PSU-resampling replicates.")
    intervals = samples.drop(columns="replicate").quantile([0.025, 0.975]).T
    intervals.columns = ["lower_psu", "upper_psu"]
    return samples, intervals


def run(source: Path, output: Path, replicates: int) -> None:
    lock = verify_frozen_objects()
    raw = read_data(source)
    frame, flow, design_universe = prepare(raw)
    if frame.diabetes.nunique() != 2:
        raise RuntimeError("Eligible Jordan sample does not contain both outcome classes.")
    if len(frame) < 100 or frame.diabetes.sum() < 20:
        raise RuntimeError("Eligible Jordan sample is too small for the prespecified primary evaluation.")
    output.mkdir(parents=True, exist_ok=True)
    flow.to_csv(output / "jordan_cohort_flow.csv", index=False)

    core = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "bmi", "waist_to_hip_ratio", "hypertension_history"]
    missing = frame[core].isna().sum().rename("missing_n").to_frame()
    missing["missing_percent"] = missing.missing_n / len(frame) * 100
    missing.to_csv(output / "jordan_predictor_missingness.csv", index_label="variable")

    predictions = frame[["id", "nationality", "diabetes", "analysis_weight", "design_stratum", "design_psu"]].copy()
    probabilities = {}
    rows = []
    for config in CONFIGS:
        obj = joblib.load(MODEL_DIR / f"{config}_LogReg_frozen.joblib")
        if obj.get("external_refitting_permitted") is not False or obj.get("external_recalibration_permitted") is not False:
            raise RuntimeError(f"Frozen policy metadata invalid for {config}.")
        score = obj["model"].decision_function(obj["preprocessor"].transform(frame[obj["features"]]))
        probability = obj["calibrator"].predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]
        probabilities[config] = probability
        predictions[f"{config}_probability"] = probability
        rows.append(
            {
                "config": config,
                "n": len(frame),
                "cases": int(frame.diabetes.sum()),
                "threshold": float(obj["threshold"]),
                **metric_function(frame.diabetes, probability, frame.analysis_weight, obj["threshold"]),
            }
        )

    results = pd.DataFrame(rows)
    threshold = float(joblib.load(MODEL_DIR / "BMI_LogReg_frozen.joblib")["threshold"])
    samples, intervals = resample_intervals(
        frame, probabilities, threshold, replicates, design_universe
    )
    results.to_csv(output / "jordan_external_validation_results.csv", index=False)
    intervals.to_csv(output / "jordan_external_validation_psu_intervals.csv", index_label="metric")
    samples.to_csv(output / "jordan_external_validation_resampling_replicates.csv", index=False)
    predictions.to_csv(output / "jordan_external_validation_predictions.csv", index=False)

    complete = frame[core].notna().all(axis=1)
    complete_rows = []
    for config, probability in probabilities.items():
        complete_rows.append(
            {
                "config": config,
                "n": int(complete.sum()),
                "cases": int(frame.loc[complete, "diabetes"].sum()),
                **metric_function(
                    frame.loc[complete, "diabetes"],
                    probability[complete.to_numpy()],
                    frame.loc[complete, "analysis_weight"],
                    threshold,
                ),
            }
        )
    pd.DataFrame(complete_rows).to_csv(output / "jordan_complete_case_sensitivity.csv", index=False)

    nationality_rows = []
    nationality_manifest = []
    for nationality, subgroup in frame.groupby("nationality", sort=True):
        subgroup_mask = frame.index.isin(subgroup.index)
        cases = int(subgroup.diabetes.sum())
        noncases = int((subgroup.diabetes == 0).sum())
        status = "completed" if len(subgroup) >= 100 and cases >= 20 and noncases >= 20 else "insufficient"
        nationality_manifest.append(
            {
                "nationality": str(nationality),
                "status": status,
                "n": int(len(subgroup)),
                "cases": cases,
                "noncases": noncases,
                "n_strata": int(subgroup.design_stratum.nunique()),
                "n_psus": int(subgroup.design_psu.nunique()),
            }
        )
        if status != "completed":
            continue
        subgroup_probabilities = {
            config: probability[subgroup_mask] for config, probability in probabilities.items()
        }
        for config, probability in subgroup_probabilities.items():
            nationality_rows.append(
                {
                    "nationality": str(nationality),
                    "config": config,
                    "n": int(len(subgroup)),
                    "cases": cases,
                    "threshold": threshold,
                    **metric_function(
                        subgroup.diabetes,
                        probability,
                        subgroup.analysis_weight,
                        threshold,
                    ),
                }
            )
        subgroup_universe = design_universe.loc[
            design_universe.nationality.astype(str).eq(str(nationality))
        ].copy()
        group_samples, group_intervals = resample_intervals(
            subgroup,
            subgroup_probabilities,
            threshold,
            replicates,
            subgroup_universe,
        )
        slug = str(nationality).strip().lower().replace(" ", "_")
        group_samples.to_csv(
            output / f"jordan_{slug}_resampling_replicates.csv", index=False
        )
        group_intervals.to_csv(
            output / f"jordan_{slug}_psu_intervals.csv", index_label="metric"
        )
    pd.DataFrame(nationality_rows).to_csv(
        output / "jordan_nationality_sensitivity.csv", index=False
    )

    manifest = {
        "source_file": source.name,
        "source_sha256": sha256(source),
        "frozen_lock_sha256": sha256(LOCK),
        "frozen_at_utc": lock["frozen_at_utc"],
        "model_objects_verified": True,
        "model_refit": False,
        "recalibration": False,
        "threshold_optimization": False,
        "synthetic_data_used": False,
        "outcome": "h7a=1 versus h7a=2 among h6=1; related but nonidentical to NHANES DIQ010",
        "analysis_weight": "wstep2",
        "resampling_replicates_requested": replicates,
        "resampling_replicates_valid": len(samples),
        "n_strata": int(frame.design_stratum.nunique()),
        "n_psus": int(frame.design_psu.nunique()),
        "resampling_design_universe": "all public-file participants with positive wstep2; metrics restricted to the eligible h6=1 outcome domain",
        "resampling_universe_n_strata": int(design_universe.design_stratum.nunique()),
        "resampling_universe_n_psus": int(design_universe.design_psu.nunique()),
        "nationality_sensitivity": nationality_manifest,
    }
    (output / "jordan_external_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(flow.to_string(index=False))
    print(results[["config", "n", "cases", "auc", "brier", "calibration_slope", "calibration_in_the_large"]].to_string(index=False))
    print(f"Results written to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("participant_file", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "jordan_results")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    run(args.participant_file.resolve(), args.output.resolve(), args.replicates)

