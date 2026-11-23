"""Reviewer-directed reanalysis. Run audit first; no original file is modified.

The old test observations have already been inspected in the submitted study.
Reanalysis results are internal, post-review estimates, not a new external test.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'data'
OUT = ROOT / 'outputs'
ZERO_ARTIFACT = 5.397605346934028e-79
SEED = 42
NUMERIC = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
           'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
           'triglycerides', 'physical_activity_minutes_per_week',
           'alcohol_drinks_per_drinking_day', 'sleep_hours_per_day',
           'poverty_income_ratio', 'glucose_fasting']
CATEGORICAL = ['ever_smoked_100_cigarettes', 'education_level', 'ethnicity',
               'hypertension_history', 'family_history_diabetes']
CONFIGS = {'BASE': ['bmi', 'waist_to_hip_ratio'], 'BMI': ['waist_to_hip_ratio'],
           'WHR': ['bmi'], 'BOTH': []}
BASE_PANELS = ['mec_nonfasting', 'fasting_with_glucose', 'fasting_without_glucose']
PANELS = BASE_PANELS + [x+'_borderline_nonpositive' for x in BASE_PANELS]


def dump(path, obj):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
        if isinstance(value, np.generic): return value.item()
        return value
    Path(path).write_text(json.dumps(clean(obj), indent=2, default=str, allow_nan=False), encoding='utf-8')


def read_csv(path):
    # Exact known XPORT zero conversion artifact, not a generic small-value cutoff.
    return pd.read_csv(path, float_precision='round_trip').replace(ZERO_ARTIFACT, 0.0)


def load_data():
    import design
    raw,data=design.reconstruct()
    return raw,data[data.DIQ010.isin([1,2])].copy()


def get_panel(data,panel):
    import design
    d=design.load(panel).drop(columns='split')
    omit=(['glucose_fasting'] if 'without_glucose' in panel else []) if panel.startswith('fasting') else ['glucose_fasting','ldl_cholesterol','triglycerides']
    return d,[c for c in NUMERIC if c not in omit]


def audit():
    import design
    facts=design.audit(PANELS)
    for panel in PANELS:
        d,nums=get_panel(None,panel)
        miss=d[nums+CATEGORICAL].isna().sum().rename('missing_n').to_frame()
        miss['missing_percent']=miss.missing_n/len(d)*100
        miss.to_csv(OUT/f'{panel}_missingness.csv',index_label='variable')
    return facts


def make_preprocessor(nums):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import KNNImputer, SimpleImputer
    # StandardScaler preserves NaNs, so distances are computed in standardized units.
    numeric = Pipeline([('scale', StandardScaler()), ('impute', KNNImputer(n_neighbors=5, keep_empty_features=True))])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numeric, nums), ('categorical', categorical, CATEGORICAL)])


def make_model(name, y, weight):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    # Retain the original fixed benchmark settings. No hyperparameter search is claimed.
    models = {
        'LogReg': lambda: LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
        'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=10000, random_state=SEED),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=2),
        'BalancedRF': lambda: BalancedRandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2),
        'XGBoost': lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.01, subsample=.8,
                                        colsample_bytree=.8, scale_pos_weight=float(weight[y == 0].sum()/weight[y == 1].sum()),
                                        random_state=SEED, eval_metric='logloss', tree_method='hist', n_jobs=2),
        'CatBoost': lambda: CatBoostClassifier(depth=4, learning_rate=.05, iterations=300,
                                               auto_class_weights='Balanced', random_seed=SEED,
                                               verbose=False, thread_count=2, allow_writing_files=False)}
    return models[name]()


def fit_model(name, X, y, w, nums):
    pre = make_preprocessor(nums)
    z = pre.fit_transform(X)
    model = make_model(name, np.asarray(y), np.asarray(w))
    model.fit(z, y, sample_weight=np.asarray(w)/np.mean(w))
    return pre, model


def score_model(pre, model, X):
    z = pre.transform(X)
    from sklearn.linear_model import LogisticRegression
    if isinstance(model,LogisticRegression):
        return model.decision_function(z),'logit'
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(z)[:, 1], 'probability'
    return model.decision_function(z), 'margin'


def calibrate(score, y, w, kind):
    from sklearn.linear_model import LogisticRegression
    x = np.asarray(score)
    if kind == 'probability':
        x = np.log(np.clip(x, 1e-6, 1-1e-6) / (1-np.clip(x, 1e-6, 1-1e-6)))
    m = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
    m.fit(x.reshape(-1, 1), y, sample_weight=np.asarray(w)/np.mean(w))
    return m


def apply_calibrator(m, score, kind):
    x = np.asarray(score)
    if kind == 'probability':
        x = np.log(np.clip(x, 1e-6, 1-1e-6) / (1-np.clip(x, 1e-6, 1-1e-6)))
    return m.predict_proba(x.reshape(-1, 1))[:, 1]


def metrics(y, p, w, threshold=.20, calibration=True):
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, confusion_matrix, precision_recall_curve, auc
    from scipy.optimize import minimize, brentq
    from scipy.special import expit
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    w = w / w.mean()
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1], sample_weight=w).ravel()
    safe = lambda a, b: float(a/b) if b else None
    pr_precision,pr_recall,_=precision_recall_curve(y,p,sample_weight=w)
    result = {'pr_auc':float(auc(pr_recall,pr_precision)), 'auc': float(roc_auc_score(y, p, sample_weight=w)),
              'brier': float(brier_score_loss(y, p, sample_weight=w)),
              'average_precision': float(average_precision_score(y, p, sample_weight=w)),
              'sensitivity': safe(tp, tp+fn), 'specificity': safe(tn, tn+fp),
              'ppv': safe(tp, tp+fp), 'npv': safe(tn, tn+fn),
              'accuracy': safe(tp+tn, tp+tn+fp+fn), 'f1': safe(2*tp, 2*tp+fp+fn)}
    if calibration:
        x = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
        def objective(beta):
            eta = beta[0] + beta[1]*x
            return np.sum(w*(np.logaddexp(0, eta)-y*eta))/w.sum()
        def gradient(beta):
            err = w*(expit(beta[0]+beta[1]*x)-y)/w.sum()
            return np.array([err.sum(), (err*x).sum()])
        opt = minimize(objective, [0., 1.], jac=gradient, method='BFGS')
        result['calibration_intercept_joint'] = float(opt.x[0]) if opt.success else None
        result['calibration_slope'] = float(opt.x[1]) if opt.success else None
        result['calibration_fit_converged'] = bool(opt.success)
        result['calibration_in_the_large'] = float(brentq(lambda a: np.sum(w*(expit(a+x)-y)), -50, 50))
    return result


def bootstrap_metrics(frame, p, replicates=500):
    """Rao-Wu rescaled PSU bootstrap within strata, conditional on fitted model.

    Does not incorporate uncertainty from model development or the random split.
    Singleton test strata cause an explicit failure rather than silent IID CIs.
    """
    rng = np.random.default_rng(SEED)
    y = frame.diabetes.to_numpy()
    w = frame.analysis_weight.to_numpy()
    strata = []
    for _, block in frame.groupby('SDMVSTRA'):
        units = block.psu_group.unique()
        if len(units) < 2:
            return pd.DataFrame(), 'Not estimated: a test stratum has only one represented PSU.'
        strata.append(units)
    records = []
    groups = frame.psu_group.to_numpy()
    for _ in range(replicates):
        mult = np.zeros(len(frame))
        for units in strata:
            m = len(units)
            chosen = rng.choice(units, m-1, replace=True)
            for unit in units:
                mult[groups == unit] = m/(m-1) * np.sum(chosen == unit)
        if len(np.unique(y[(w*mult)>0])) < 2:
            continue
        records.append(metrics(y, p, w*mult))
    b = pd.DataFrame(records)
    cols = [c for c in b if c != 'calibration_fit_converged']
    ci = b[cols].quantile([.025, .975]).T
    ci.columns = ['lower_psu', 'upper_psu']
    return ci, 'Exploratory conditional PSU-resampling interval; fixed train/validation/test partition, fitted model and calibration; three test strata, six PSUs and eight distinct patterns. Resampling quantiles 0.025 and 0.975 do not imply nominal coverage or full NHANES survey-design uncertainty.'


def save_plot(frame, probs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
    for label, p in probs.items():
        t = pd.DataFrame({'p': p, 'y': frame.diabetes.to_numpy(), 'w': frame.analysis_weight.to_numpy()})
        t['bin'] = pd.qcut(t.p, 5, duplicates='drop')
        points = [(np.average(g.p, weights=g.w), np.average(g.y, weights=g.w)) for _, g in t.groupby('bin', observed=True)]
        ax.plot(*np.array(points).T, 'o-', label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Weighted observed proportion', title='Internal test calibration')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run_panel(panel, selected_models=None, replicates=500):
    import joblib
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    _, data = load_data()
    d, nums = get_panel(data, panel)
    dest = OUT / panel; dest.mkdir(exist_ok=True, parents=True)
    import design
    assignment=design.freeze_splits()
    d=d.merge(assignment[['psu_group','split']],on='psu_group',validate='many_to_one')
    manifest=pd.read_csv(dest/'split_manifest.csv')
    assert set(d.SEQN)==set(manifest.SEQN)
    for split in ['train','validation','test']:
        assert set(d.loc[d.split==split,'SEQN'])==set(manifest.loc[manifest.split==split,'SEQN'])
    partitions=[set(d.loc[d.split==x,'psu_group']) for x in ['train','validation','test']]
    assert all(partitions[i].isdisjoint(partitions[j]) for i in range(3) for j in range(i))
    tr = d[d.split == 'train'].copy(); va = d[d.split == 'validation'].copy(); te = d[d.split == 'test'].copy()
    names = selected_models or ['LogReg', 'LinearSVM', 'RandomForest', 'BalancedRF', 'XGBoost', 'CatBoost']
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(tr, tr.diabetes, tr.SDMVSTRA.astype(str)))
    all_results, cv_rows, predictions = [], [], te[['SEQN', 'diabetes', 'analysis_weight', 'SDMVSTRA', 'SDMVPSU']].copy()
    param_manifest = {}
    for config, remove in CONFIGS.items():
        cn = [c for c in nums if c not in remove]; feats = cn + CATEGORICAL
        fitted, val_scores = {}, {}
        for name in names:
            scores = []
            for fold, (a, b) in enumerate(cv):
                aa, bb = tr.iloc[a], tr.iloc[b]
                assert set(aa.psu_group).isdisjoint(bb.psu_group)
                pre, model = fit_model(name, aa[feats], aa.diabetes, aa.analysis_weight, cn)
                s, _ = score_model(pre, model, bb[feats])
                score = float(roc_auc_score(bb.diabetes, s, sample_weight=bb.analysis_weight))
                scores.append(score)
                cv_rows.append(dict(panel=panel, config=config, model=name, fold=fold, weighted_auc=score))
            pre, model = fit_model(name, tr[feats], tr.diabetes, tr.analysis_weight, cn)
            s, kind = score_model(pre, model, va[feats])
            fitted[name] = (pre, model, kind)
            val_scores[name] = s
            all_results.append(dict(config=config, model=name, cv_auc_mean=float(np.mean(scores)),
                                    cv_auc_sd=float(np.std(scores, ddof=1)),
                                    validation_auc=float(roc_auc_score(va.diabetes, s, sample_weight=va.analysis_weight))))
            param_manifest[f'{config}/{name}'] = model.get_params()
            pd.DataFrame(cv_rows).to_csv(dest/'cv_folds.csv', index=False)
            print(f'{panel} {config} {name} CV={np.mean(scores):.4f}', flush=True)
        comparison = pd.DataFrame([r for r in all_results if r['config'] == config])
        winner = comparison.sort_values(['cv_auc_mean', 'model'], ascending=[False, True]).iloc[0]['model']
        # Fixed LR comparison separates anthropometric contribution from algorithm selection.
        evaluate = list(dict.fromkeys([winner] + (['LogReg'] if 'LogReg' in names else [])))
        for name in evaluate:
            pre, model, kind = fitted[name]
            cal = calibrate(val_scores[name], va.diabetes, va.analysis_weight, kind)
            s, _ = score_model(pre, model, te[feats])
            p = apply_calibrator(cal, s, kind)
            label = f'{config}_{name}'
            predictions[label+'_calibrated'] = p
            probabilities = {'Recalibrated': p}
            variants = [('calibrated', p)]
            if kind in ['probability','logit']:
                from scipy.special import expit
                raw=s if kind=='probability' else expit(s)
                predictions[label+'_raw'] = raw
                probabilities['Uncalibrated'] = raw
                variants.append(('raw', raw))
            for variant, probability in variants:
                row = dict(config=config, model=name, variant=variant, selected_by_cv=(name==winner),
                           train_n=len(tr), validation_n=len(va), test_n=len(te), test_cases=int(te.diabetes.sum()),
                           **metrics(te.diabetes, probability, te.analysis_weight))
                row.update({'unweighted_'+k:v for k,v in metrics(te.diabetes, probability, np.ones(len(te)), calibration=False).items()})
                result_path = dest/f'{label}_{variant}_metrics.json'
                ci, note = bootstrap_metrics(te, probability, replicates)
                row['uncertainty_note'] = note
                row['calibration_intercept_validation'] = float(cal.intercept_[0])
                row['calibration_slope_validation'] = float(cal.coef_[0, 0])
                dump(result_path, row)
                ci.to_csv(dest/f'{label}_{variant}_psu_resampling_intervals.csv', index_label='metric')
            thresholds = [dict(threshold=t, **metrics(te.diabetes, p, te.analysis_weight, t, calibration=False)) for t in [.10, .20, .30, .50]]
            pd.DataFrame(thresholds).to_csv(dest/f'{label}_threshold_sensitivity.csv', index=False)
            joblib.dump(dict(preprocessor=pre, model=model, calibrator=cal, score_kind=kind, features=feats,
                             threshold=.20, outcome='cross-sectional self-reported diagnosed diabetes; type unspecified', external_validation=False, partition='disjoint masked PSU pairs within intact strata', borderline_nonpositive=('borderline_nonpositive' in panel), panel=panel), dest/f'{label}.joblib')
            save_plot(te, probabilities, dest/f'{label}_calibration.png')
        dump(dest/'parameters.json', param_manifest)
        pd.DataFrame(all_results).to_csv(dest/'model_development.csv', index=False)
        predictions.to_csv(dest/'test_predictions.csv', index=False,float_format='%.17g')
    results = [json.loads(p.read_text()) for p in dest.glob('*_metrics.json')]
    pd.DataFrame(results).to_csv(dest/'test_results.csv', index=False)
    dump(dest/'status.json',dict(completed=True,psu_disjoint=True,synthetic_data_used=False,panel=panel))
    print('FINISHED '+panel, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['audit', 'run'])
    parser.add_argument('--panel', choices=PANELS+['all'], default='all')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.task == 'audit':
        audit(); return
    versions = {'python': platform.python_version()}
    for package in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost','imbalanced-learn','matplotlib','joblib']:
        versions[package] = importlib.metadata.version(package)
    dump(OUT/'environment.json', versions)
    for panel in PANELS if args.panel == 'all' else [args.panel]:
        run_panel(panel, args.models, args.bootstrap)


if __name__ == '__main__':
    main()
