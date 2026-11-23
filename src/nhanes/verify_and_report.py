from pathlib import Path
import json,numpy as np,pandas as pd,hashlib
from sklearn.metrics import roc_auc_score,brier_score_loss
import revision_analysis as ra

def md(frame):return frame.to_markdown(index=False,floatfmt='.4f')
def run():
    rows=[];checks=[];neural_rows=[]
    for panel in ra.PANELS:
        dest=ra.OUT/panel;m=pd.read_csv(dest/'split_manifest.csv');pred=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip');results=pd.read_csv(dest/'test_results.csv')
        sets=[set(m.loc[m.split==x,'psu_group']) for x in ['train','validation','test']]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
        assert set(pred.SEQN)==set(m.loc[m.split=='test','SEQN'])
        for row in results.itertuples():
            p=pred[f'{row.config}_{row.model}_{row.variant}'];assert np.isclose(roc_auc_score(pred.diabetes,p,sample_weight=pred.analysis_weight),row.auc,atol=1e-10)
            assert np.isclose(brier_score_loss(pred.diabetes,p,sample_weight=pred.analysis_weight),row.brier,atol=1e-10)
        lr=results[(results.model=='LogReg')&(results.variant=='calibrated')].copy();lr.insert(0,'panel',panel);rows.append(lr)
        status=json.loads((dest/'neural/status.json').read_text());assert status['completed']
        nr=pd.read_csv(dest/'neural/test_results.csv');development=pd.read_csv(dest/'neural/development_by_seed.csv');assert len(development)==60
        development.groupby(['config','model']).agg(validation_auc_mean=('validation_auc','mean'),validation_auc_sd=('validation_auc','std'),median_epoch=('best_epoch','median')).reset_index().to_csv(dest/'neural/development_summary.csv',index=False)
        for row in nr.itertuples():
            pr=pd.read_csv(dest/'neural'/f'{row.config}_test_predictions.csv',float_precision='round_trip')
            assert set(pr.SEQN)==set(pred.SEQN)
            assert np.isclose(roc_auc_score(pr.outcome,pr[row.variant],sample_weight=pr.weight),row.auc,atol=1e-10)
            assert np.isclose(brier_score_loss(pr.outcome,pr[row.variant],sample_weight=pr.weight),row.brier,atol=1e-10)
        nr.insert(0,'panel',panel);neural_rows.append(nr)
        checks.append({'panel':panel,'psu_overlap':0,'classical_metric_rows_verified':len(results),'neural_metric_rows_verified':len(nr),'neural_development_fits':len(development),'final_neural_fits':4})
    lr=pd.concat(rows,ignore_index=True);lr.to_csv(ra.OUT/'common_logistic_results.csv',index=False)
    pd.concat(neural_rows,ignore_index=True).to_csv(ra.OUT/'all_neural_test_results.csv',index=False)
    pair=pd.read_csv(ra.OUT/'paired_logistic_auc_differences.csv');conclusions=[]
    for panel in ra.BASE_PANELS:
        for scenario,panel_id in [('borderline excluded',panel),('borderline non-positive',panel+'_borderline_nonpositive')]:
            row=pair[(pair.panel==panel_id)&(pair.comparison=='WHR minus BMI')].iloc[0]
            interpretation='interval includes zero'
            if row.lower_psu>0:interpretation='WHR higher within this internal comparison'
            if row.upper_psu<0:interpretation='BMI higher within this internal comparison'
            conclusions.append(dict(panel=panel,scenario=scenario,delta_auc=row.delta_auc,lower_psu=row.lower_psu,upper_psu=row.upper_psu,interpretation=interpretation))
    conclusions=pd.DataFrame(conclusions);conclusions.to_csv(ra.OUT/'borderline_conclusion_sensitivity.csv',index=False)
    # Evaluate the sensitivity-fitted models on the original definite-response test people too.
    fixed=[]
    for panel in ra.BASE_PANELS:
        original=pd.read_csv(ra.OUT/panel/'test_predictions.csv',float_precision='round_trip')
        added=pd.read_csv(ra.OUT/(panel+'_borderline_nonpositive')/'test_predictions.csv',float_precision='round_trip').set_index('SEQN').loc[original.SEQN]
        for config in ra.CONFIGS:
            p=added[f'{config}_LogReg_calibrated'].to_numpy()
            fixed.append(dict(panel=panel,config=config,**ra.metrics(original.diabetes,p,original.analysis_weight)))
    pd.DataFrame(fixed).to_csv(ra.OUT/'borderline_models_on_definite_test.csv',index=False)
    hashes=json.loads((ra.OUT/'source_sha256.json').read_text());assert all(hashlib.sha256((ra.SOURCE/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    ra.dump(ra.OUT/'complete_verification.json',{'checks':checks,'source_hashes_unchanged':True,'synthetic_data_used':False,'documents_generated':False})
    composition=pd.read_csv(ra.OUT/'cohort_and_split_composition.csv')
    sections=['# Corrected PSU partition results','These results use newly frozen, disjoint masked PSU partitions. The primary analysis is the examination-weighted nonfasting panel. All synthetic-data analyses have been removed. This is cross-sectional classification of self-reported diagnosed diabetes; DIQ010 does not determine diabetes type.','## Cohort and split composition',md(composition),'## Common logistic models after validation calibration',md(lr[['panel','config','auc','brier','sensitivity','specificity','calibration_in_the_large','calibration_intercept_joint','calibration_slope','average_precision','pr_auc']]),'## Paired BMI versus WHR and borderline outcome sensitivity',md(conclusions),'## All paired anthropometric differences',md(pair),'## Complete case sensitivity',md(pd.read_csv(ra.OUT/'complete_case_sensitivity.csv')),'## Neural model estimates',md(pd.concat(neural_rows)[['panel','config','model','variant','auc','brier','specificity','calibration_slope']]),'## Interpretation limits','The test set contains three masked strata and six PSUs. Exploratory conditional PSU-resampling intervals are conditional on the fixed train/validation/test partition, fitted models and these held-out strata, exclude development and partition uncertainty, and have only eight distinct within-stratum resampling patterns. They do not represent full NHANES survey-design uncertainty or all US adult males, and no nominal confidence coverage is claimed. The study remains internal and post-review. Contemporaneous glucose and other measurements may reflect existing diagnosis, treatment, or behavioral changes after diagnosis. No prospective or clinical utility claim follows from the AUC. No suitable real independent cohort with harmonized predictors was available. Genuine external validation requires an unchanged frozen NHANES model, preprocessing, calibration procedure, and decision threshold applied to real independent data.','## Revision record purpose','The previous Scientific Reports submission was rejected. The reviewer-response record is for internal revision documentation or a possible appeal, not an ordinary revision submission.']
    (ra.ROOT/'Corrected_results.md').write_text('\n\n'.join(sections),encoding='utf-8')
    print(conclusions.to_string(index=False),flush=True)
if __name__=='__main__':run()
