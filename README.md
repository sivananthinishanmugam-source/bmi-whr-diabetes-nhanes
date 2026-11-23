# Survey-Weighted Comparison of BMI and Waist-to-Hip Ratio for Classifying Self-Reported Diagnosed Diabetes in Adult Men

Code, public data, and aggregate outputs for the manuscript:

> Shanmugam SN, Okpete UE, Mozumder MAI, Kim H-C. *Survey-weighted comparison of body mass index and waist-to-hip ratio for classification of self-reported diagnosed diabetes in adult men: a cross-sectional study with internal evaluation in NHANES and external validation of harmonized reduced models in Jordan STEPS.*

---

## 1. Description

This repository compares body mass index (BMI) and waist-to-hip ratio (WHR) for cross-sectional classification of self-reported diagnosed diabetes among adult men. It combines:

- **matched anthropometric configurations**: BASE (neither measure), BMI, WHR, and BMI + WHR, with all other predictors held fixed;
- **survey-design-aware evaluation**: training, validation, and test partitions separated at the masked primary sampling unit (PSU) level, with NHANES examination and fasting survey weights used in fitting and evaluation;
- **classical and neural models**: logistic regression, linear SVM, random forest, balanced random forest, XGBoost, CatBoost, multilayer perceptron, Wide & Deep, and residual networks;
- **validation-only calibration**, **SHAP explanations**, and **conditional PSU-resampling intervals**;
- **locked-model external validation**: four reduced logistic models frozen before access to Jordan STEPS 2019 participant data and applied unchanged.

All preprocessing (standardization, KNN imputation, one-hot encoding) is fitted on the training partition only, so no information from validation or test data leaks into model development.

In the primary NHANES analysis (2,589 men; 459 positive reports), the BMI + WHR logistic model reached ROC-AUC 0.8863 (Brier score 0.0866). The paired WHR-minus-BMI AUC difference was 0.0063 (-0.0034 to 0.0149), and adding WHR to BMI changed AUC by 0.0029 (-0.0039 to 0.0107). In Jordan STEPS 2019 (885 men; 249 positive), the frozen models reached AUC 0.7617 (BMI) and 0.7378 (WHR); the BMI-minus-WHR difference was 0.0239 (-0.0018 to 0.0559).

## 2. Dataset Information

| Item | Details |
|------|---------|
| Development data | National Health and Nutrition Examination Survey (NHANES) 2017-2018, National Center for Health Statistics (CDC) |
| Public source | https://wwwn.cdc.gov/nchs/nhanes/ (file links in `data/DATA_SOURCES.md`) |
| Files in this repository | `data/raw/` (14 NHANES XPORT files) and `data/prepared/NHANES_2017_2018_Merged.csv` (merged by `src/nhanes/prepare_raw.py`) |
| Population | Adult men aged 18 years or older; primary panel 2,589 men (459 positive reports), fasting panel 1,096 men (213 positive reports) |
| Outcome | NHANES DIQ010: Yes (1) positive, No (2) non-positive; borderline (3) excluded in the primary analysis and coded non-positive in a sensitivity analysis; type 1 and type 2 diabetes are not distinguished |
| External validation data | Jordan STEPS 2019, WHO NCD Microdata Repository (catalog 853): https://extranet.who.int/ncdsmicrodata/index.php/catalog/853 |
| Jordan data access | Licensed participant-level data; **not included**. Authorized users place `jor2019.dta` and `jor2019.csv` in `src/jordan_external_validation/raw/` |
| Privacy | Public or licensed de-identified secondary data; no new participant data were collected |

## 3. Code Information

```
data/          NHANES raw and prepared files; DATA_SOURCES.md
notebooks/     Jupyter notebooks, numbered in execution order
outputs/
  nhanes/      Saved aggregate NHANES results used for checking
  jordan/      Saved aggregate Jordan STEPS results used for checking
src/
  nhanes/                       NHANES pipeline: preparation, PSU design, models, calibration, resampling, SHAP
  jordan_external_validation/   Model freezing, harmonization protocol, external validation and verification
figures/       Manuscript Figures 1-5
tables/        Manuscript Tables 1-4 (CSV)
```

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_acquisition_and_provenance.ipynb` | Source files, data provenance, and the pre-access model-freeze lock |
| 02 | `02_nhanes_pipeline.ipynb` | Optional full NHANES rerun from raw data (`RUN_NHANES_PIPELINE = True`) |
| 03 | `03_nhanes_analysis.ipynb` | NHANES cohort, PSU partition, preprocessing, models, paired contrasts, and SHAP, step by step |
| 04 | `04_cohorts_psu_and_models.ipynb` | Cohorts, PSU separation, and primary and secondary model results (Tables 1-2) |
| 05 | `05_outcome_and_sensitivity_analyses.ipynb` | Outcome coding and sensitivity analyses (Table 3) and SHAP importance |
| 06 | `06_jordan_external_validation.ipynb` | Jordan STEPS external validation of the frozen reduced models, step by step |
| 07 | `07_jordan_validation_results.ipynb` | External performance and paired contrasts (Table 4) |
| 08 | `08_verify_reported_results.ipynb` | Automatic checks of the values reported in the manuscript (prints PASS) |

## 4. Requirements

- Python 3.11
- Packages (see `requirements.txt`; versions used in the study are pinned):
  - numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0
  - xgboost 3.2.0, catboost 1.2.10, imbalanced-learn 0.14.1
  - tensorflow 2.21.0, keras 3.14.0, shap 0.51.0, matplotlib 3.10.8, joblib 1.5.3
  - nbformat, nbclient, ipykernel (notebook execution)

## 5. Usage Instructions

1. **Get the code**: download or clone this repository.
2. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. **Data**: the NHANES files are already included in `data/`. To download them again from CDC:
   ```bash
   python src/nhanes/download_sources.py
   python src/nhanes/prepare_raw.py
   ```
4. **Run the notebooks** in numerical order from the `notebooks/` folder:
   ```bash
   jupyter notebook notebooks/
   ```
   - **Check the reported results:** run 04, 05, 07, and 08. They compare the saved results in `outputs/` with the manuscript values; notebook 08 ends with PASS lines.
   - **Full NHANES rerun:** set `RUN_NHANES_PIPELINE = True` in notebook 02, or run `python src/nhanes/run_all.py --analysis-only`. The neural benchmarks make this slow. New results are written to `src/nhanes/outputs/`, and notebooks 04-08 then check those instead of the saved results.
   - **Jordan STEPS rerun (licensed data required):**
     ```bash
     python src/jordan_external_validation/external_validation/run_jordan_external_validation.py src/jordan_external_validation/raw/jor2019.dta --output src/jordan_external_validation/external_validation/jordan_results --replicates 2000
     python src/jordan_external_validation/external_validation/verify_jordan_external_validation.py
     ```
   - **Refit the frozen reduced models for comparison** (does not overwrite the submitted models):
     ```bash
     python src/jordan_external_validation/external_validation/freeze_nhanes_transport_model.py --output my_refit
     ```

All notebook paths are relative to the repository folder.

## 6. Methodology

1. **Cohort:** NHANES 2017-2018 men aged 18 years or older with a definite DIQ010 response and a positive examination (WTMEC2YR) or fasting (WTSAF2YR) weight.
2. **Partitioning:** the 15 masked strata (30 PSUs) assigned as whole strata to training (9 strata), validation (3), and testing (3), with no participant or PSU overlap; the assignment is frozen in `src/nhanes/protocol/`.
3. **Analysis hierarchy:** primary nonfasting examination-weighted analysis; secondary fasting analysis with measured glucose; sensitivity analyses without glucose and with borderline responses coded non-positive.
4. **Preprocessing (training data only):** z-score standardization, five-neighbour KNN imputation, mode imputation, and one-hot encoding; no participant excluded for missing BMI or WHR.
5. **Models:** fixed-setting classical models selected by survey-weighted three-fold cross-validation grouped by strata; neural benchmarks with validation-based epoch selection; logistic regression retained as the common comparison model.
6. **Calibration and evaluation:** sigmoid recalibration fitted in the validation partition only; held-out test ROC-AUC, average precision, PR AUC, Brier score, calibration-in-the-large, calibration slope, and fixed-threshold summaries.
7. **Uncertainty:** exploratory conditional PSU-resampling intervals for metrics and paired AUC differences.
8. **External validation:** four reduced logistic models (age, blood pressure, heart rate, hypertension history, with BMI and/or WHR) developed in NHANES men aged 18-69 years, frozen with SHA-256 hashes before access to Jordan STEPS 2019 data, and applied without refitting, recalibration, or threshold changes.
9. **Explainability:** linear SHAP for the calibrated BMI + WHR logistic model.

## 7. Citations

If you use this code, please cite the article above.

Data:

> National Center for Health Statistics. National Health and Nutrition Examination Survey 2017-2018. Centers for Disease Control and Prevention. https://wwwn.cdc.gov/nchs/nhanes/
>
> World Health Organization. Jordan - STEPS 2019. NCD Microdata Repository. https://extranet.who.int/ncdsmicrodata/index.php/catalog/853

## 8. License & Contribution Guidelines

- **License:** the code is released under the MIT License (see `LICENSE`). NHANES data are public; Jordan STEPS 2019 data remain subject to the WHO licence conditions and are not redistributed.
- **Contributions:** bug reports and suggestions are welcome through GitHub Issues. To propose a change, fork the repository, create a branch, and open a pull request that confirms notebook 08 still reports PASS.

## Contact

Corresponding author: Hee-Cheol Kim (heeki@inje.ac.kr), Institute of Digital Anti-Aging HealthCare, Inje University, Gimhae, Republic of Korea.
