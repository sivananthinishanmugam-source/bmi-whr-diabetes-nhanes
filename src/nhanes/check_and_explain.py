"""Validate saved predictions and quantify paired anthropometric differences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss
import revision_analysis as ra


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    checks=[];deltas=[]
    _,data=ra.load_data()
    for panel in ra.PANELS:
        dest=ra.OUT/panel
        pred=pd.read_csv(dest/'test_predictions.csv')
        results=pd.read_csv(dest/'test_results.csv')
        manifest=pd.read_csv(dest/'split_manifest.csv')
        assert not manifest.SEQN.duplicated().any()
        assert set(pred.SEQN)==set(manifest.loc[manifest.split=='test','SEQN'])
        for row in results.itertuples():
            key=f'{row.config}_{row.model}_{row.variant}'
            p=pred[key].to_numpy();y=pred.diabetes.to_numpy();w=pred.analysis_weight.to_numpy()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            assert np.isclose(roc_auc_score(y,p,sample_weight=w),row.auc)
            assert np.isclose(brier_score_loss(y,p,sample_weight=w),row.brier)
            if not row.calibration_fit_converged:raise ValueError('Calibration fit needs review: '+key)
        checks.append(dict(panel=panel,metrics_verified=len(results),unique_test_ids=len(pred),passed=True))
        # Paired differences on identical people with a common algorithm and bootstrap weights.
        rng=np.random.default_rng(ra.SEED)
        pairs=[('BMI','BASE'),('WHR','BASE'),('BOTH','BASE'),('WHR','BMI'),('BOTH','BMI')]
        for a,b in pairs:
            pa=pred[f'{a}_LogReg_calibrated'].to_numpy();pb=pred[f'{b}_LogReg_calibrated'].to_numpy()
            y=pred.diabetes.to_numpy();w=pred.analysis_weight.to_numpy()
            point=roc_auc_score(y,pa,sample_weight=w)-roc_auc_score(y,pb,sample_weight=w)
            strata=[g for _,g in pred.groupby('SDMVSTRA')]
            singleton=any(g.SDMVPSU.nunique()<2 for g in strata)
            reps=[]
            if not singleton:
                for _ in range(1000):
                    mult=np.zeros(len(pred))
                    for g in strata:
                        units=g.SDMVPSU.unique();m=len(units);selected=rng.choice(units,m-1,replace=True)
                        for unit in units:
                            mult[g.index[g.SDMVPSU==unit]]=m/(m-1)*sum(selected==unit)
                    bw=w*mult
                    if len(np.unique(y[bw>0]))<2: continue
                    reps.append(roc_auc_score(y,pa,sample_weight=bw)-roc_auc_score(y,pb,sample_weight=bw))
            ci=np.quantile(reps,[.025,.975]) if reps else [None,None]
            deltas.append(dict(panel=panel,comparison=a+' minus '+b,delta_auc=point,lower_psu=ci[0],upper_psu=ci[1],replicates=len(reps)))
        # Explain the common-algorithm BOTH benchmark, not whichever model has the best test AUC.
        package=joblib.load(dest/'BOTH_LogReg.joblib')
        d,_=ra.get_panel(data,panel);d=d.merge(manifest[['SEQN','split']],on='SEQN',validate='one_to_one')
        tr=d[d.split=='train'];te=d[d.split=='test']
        pre=package['preprocessor'];model=package['model'];feats=package['features']
        assert not {'diabetes','DIQ010','SEQN','analysis_weight','SDMVSTRA','SDMVPSU'}.intersection(feats)
        ztr=pre.transform(tr[feats]);zte=pre.transform(te[feats])
        explanation=shap.LinearExplainer(model,shap.maskers.Independent(ztr,max_samples=100))(zte)
        # Calibration is affine in raw log-odds; transform SHAP and baseline exactly.
        slope=float(package['calibrator'].coef_[0,0]);intercept=float(package['calibrator'].intercept_[0])
        values=explanation.values*slope
        base=explanation.base_values*slope+intercept
        expected_logit=intercept+slope*model.decision_function(zte)
        assert np.allclose(base+values.sum(axis=1),expected_logit,atol=1e-5)
        from scipy.special import expit
        assert package['score_kind']=='logit'
        assert np.allclose(expit(expected_logit),pred['BOTH_LogReg_calibrated'],atol=1e-10)
        encoded=pre.get_feature_names_out()
        aggregated={}
        for i,name in enumerate(encoded):
            rawname=name.split('__',1)[1]
            original=next((c for c in ra.CATEGORICAL if rawname.startswith(c+'_')),rawname)
            aggregated[original]=aggregated.get(original,np.zeros(len(te)))+values[:,i]
        importance=pd.DataFrame([{'feature':k,'weighted_mean_absolute_shap':np.average(np.abs(v),weights=te.analysis_weight)} for k,v in aggregated.items()]).sort_values('weighted_mean_absolute_shap',ascending=False)
        importance.to_csv(dest/'BOTH_LogReg_shap_importance.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,6));view=importance.head(15).iloc[::-1]
        ax.barh(view.feature,view.weighted_mean_absolute_shap,color='#38698a')
        ax.set(xlabel='Weighted mean absolute SHAP value (calibrated log-odds)',title='Logistic regression with BMI and WHR')
        fig.tight_layout();fig.savefig(dest/'BOTH_LogReg_shap.png',dpi=180);plt.close(fig)
        ra.dump(dest/'shap_scope.json',dict(model='BOTH LogReg',output_scale='validation-calibrated log-odds',
                                           background='up to 100 training observations, unweighted masker',
                                           importance_aggregation='survey-weighted test mean absolute contribution',
                                           explanation='Interventional linear SHAP; not causal; dummy contributions summed by source predictor',
                                           additivity_verified=True))
        print('Verified and explained',panel,flush=True)
    pd.DataFrame(deltas).to_csv(ra.OUT/'paired_logistic_auc_differences.csv',index=False)
    ra.dump(ra.OUT/'verification.json',checks)


if __name__=='__main__':
    deps=[ra.OUT/p/'test_results.csv' for p in ra.PANELS]+[ra.OUT/p/'BOTH_LogReg.joblib' for p in ra.PANELS]+[Path(__file__)]
    check=ra.OUT/'verification.json'
    if check.exists() and check.stat().st_mtime>max(x.stat().st_mtime for x in deps):
        print('Current classical predictions, paired comparisons, and SHAP already verified.')
    else:run()
