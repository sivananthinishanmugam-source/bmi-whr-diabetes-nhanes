# Prespecified protocol for genuine external validation

## Status

This protocol was written before access to, or analysis of, participant-level Jordan STEPS 2019 data. The Jordan survey is a real, independently collected national probability sample. The synthetic Kaggle data are excluded from this workstream.

The previously fitted full NHANES models cannot be applied directly because Jordan STEPS does not contain harmonized versions of every predictor. A reduced transportable model will therefore be developed in the existing NHANES training partition, calibrated in the existing NHANES validation partition, frozen, and checked once in the existing NHANES test partition. Its preprocessing transformations, coefficients, calibration equation, and decision threshold will then be applied unchanged to Jordan STEPS.

## Target estimand and populations

- **Development population:** NHANES 2017-2018 men aged 18-69 years with `DIQ010` equal to 1 (yes) or 2 (no), a positive MEC examination weight, and the already frozen PSU-level train/validation/test assignment.
- **External population:** Jordan STEPS 2019 men aged 18-69 years who participated in Step 2, have a positive Step 2 analysis weight, and have a nonmissing response to `h7a`.
- **NHANES outcome:** cross-sectional self-report that a health professional diagnosed diabetes (`DIQ010=1`). `DIQ010` does not distinguish type 1 from type 2 diabetes.
- **Jordan outcome:** cross-sectional self-report that a doctor or other health worker said the participant had raised blood sugar or diabetes (`h7a=1`). This is broader than the NHANES wording and was asked only of participants reporting that their blood sugar had previously been measured (`h6=1`). It is therefore a related but nonidentical outcome in a selected subgroup.

The external exercise evaluates transportability of cross-sectional classification. It is not prospective prediction, screening validation, or evidence of preventive or clinical utility.

## Frozen transportable predictor set

| Analysis variable | NHANES construction | Jordan STEPS construction | Decision |
|---|---|---|---|
| Age (years) | `RIDAGEYR` | `age` | Include; restrict both datasets to 18-69 years. |
| BMI (kg/m2) | `BMXBMI` | `m12/(m11/100)^2` | Include in BMI and BOTH configurations. |
| Waist-to-hip ratio | `BMXWAIST/BMXHIP` | `m14/m15` | Include in WHR and BOTH configurations. |
| Systolic BP (mmHg) | Mean of available `BPXSY1`-`BPXSY4`, requiring at least two readings | Mean of valid `m4a`, `m5a`, `m6a`, requiring at least two readings | Include. |
| Diastolic BP (mmHg) | Mean of available `BPXDI1`-`BPXDI4`, requiring at least two readings; values below 30 excluded | Mean of valid `m4b`, `m5b`, `m6b`, requiring at least two readings | Include. |
| Heart rate (beats/min) | `BPXPLS` | Mean of valid `m16a`, `m16b`, `m16c`, requiring at least two readings | Include. |
| Hypertension history | `BPQ020`: yes/no | `h2a`: yes/no if BP was previously measured | Include; missing responses remain missing and are imputed using the frozen NHANES preprocessing rule. |

Four prespecified logistic-regression configurations will be frozen: BASE (neither BMI nor WHR), BMI, WHR, and BOTH. All use the same core covariates. The primary external comparison is BMI versus WHR discrimination; BOTH is descriptive.

## Variables deliberately excluded from the external model

- Ethnicity, poverty-income ratio, sleep, and family history of diabetes are absent or not harmonized.
- Education and income categories are country-specific and are not forced into NHANES categories.
- Jordan tobacco questions do not reproduce the NHANES lifetime threshold of 100 cigarettes.
- Alcohol and physical-activity items are similar but differ in skip patterns and derivation details; they are omitted from the strict core model.
- Total cholesterol, HDL, LDL, triglycerides, and fasting glucose require Step 3 participation. They are omitted from the primary external model to preserve the larger Step 2 target sample and avoid treating contemporaneous glucose as prospective information.

## Frozen development procedure

1. Retain the existing disjoint NHANES PSU-level train, validation, and test partitions.
2. Fit the preprocessing steps only in the training partition: standardized numeric variables followed by five-nearest-neighbor imputation, and most-frequent imputation plus one-hot encoding for hypertension history.
3. Fit survey-weighted, class-balanced logistic regression in the training partition.
4. Fit a logistic calibration equation in the NHANES validation partition.
5. Freeze a probability threshold of 0.20, matching the revised internal analysis.
6. Save the complete preprocessing-model-calibration objects, schema, hashes, and creation log before participant-level Jordan data are accessed.
7. Apply every frozen object once to Jordan without refitting, recentering, recalibration, threshold optimization, or feature selection.

## External evaluation

Report unweighted counts and Step 2 weighted estimates for AUC, Brier score, calibration-in-the-large, calibration slope, sensitivity, specificity, PPV, NPV, accuracy, and F1 at the frozen 0.20 threshold. Estimate uncertainty by resampling PSUs within Jordan strata while keeping predictions and all fitted objects fixed. Report the number of strata and PSUs contributing to the analysis and fail explicitly if the public file cannot support the intended design-based resampling.

The main BMI-versus-WHR conclusion will be based on the paired weighted AUC difference (`AUC_BMI - AUC_WHR`) with PSU-resampling limits. Calibration will be interpreted cautiously because the Jordan outcome wording and eligible denominator differ from NHANES.

## Prespecified sensitivity analyses

1. Complete-case analysis for the frozen core predictors.
2. Jordanian nationals and Syrian participants reported separately if sample sizes and event counts permit; no model refitting.
3. Descriptive Step 3 biochemical outcome analysis may be performed separately using fasting glucose and diabetes medication with `wstep3`, but it will not replace the primary self-reported-outcome validation and will not use glucose as a predictor of that contemporaneous outcome.

## Access and publication conditions

The WHO catalog requires the user to acknowledge the data source, share planned publications with WHO before publication, offer coauthorship to the survey coordinator, and use the data only for noncommercial, not-for-profit public-health purposes. Participant-level access requires a WHO repository login and data request. No result will be described as completed external validation until the requested participant file has been obtained and the frozen analysis has run successfully.

