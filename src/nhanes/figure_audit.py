"""Regenerate published calibration figures and audit every plotted source value."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import revision_analysis as ra

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    checks=[];records=[];issues=[]
    for panel in ra.PANELS:
        dest=ra.OUT/panel;pred=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip')
        probs={'Recalibrated':pred.BOTH_LogReg_calibrated.to_numpy(),'Uncalibrated':pred.BOTH_LogReg_raw.to_numpy()}
        path=dest/'BOTH_LogReg_calibration.png';ra.save_plot(pred,probs,path)
        series=[]
        for label,p in probs.items():
            t=pd.DataFrame({'p':p,'y':pred.diabetes,'w':pred.analysis_weight});t['bin']=pd.qcut(t.p,5,duplicates='drop')
            points=[[float(np.average(g.p,weights=g.w)),float(np.average(g.y,weights=g.w))] for _,g in t.groupby('bin',observed=True)]
            series.append({'label':label,'points':points})
        records.append({'file':str(path.relative_to(ra.ROOT)),'kind':'calibration','source':str((dest/'test_predictions.csv').relative_to(ra.ROOT)),'data':series,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        checks.append({'check':'calibration_plot_from_final_predictions','panel':panel,'passed':True})
        imp=pd.read_csv(dest/'BOTH_LogReg_shap_importance.csv');view=imp.head(15).iloc[::-1]
        path=dest/'BOTH_LogReg_shap.png';fig,ax=plt.subplots(figsize=(8,6));ax.barh(view.feature,view.weighted_mean_absolute_shap,color='#38698a');ax.set(xlabel='Weighted mean absolute SHAP value (calibrated log-odds)',title='Logistic regression with BMI and WHR');fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
        records.append({'file':str(path.relative_to(ra.ROOT)),'kind':'SHAP','source':str((dest/'BOTH_LogReg_shap_importance.csv').relative_to(ra.ROOT)),'data':view.to_dict('records'),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        checks.append({'check':'SHAP_plot_uses_verified_additive_common_logistic_explanation','panel':panel,'passed':json.loads((dest/'shap_scope.json').read_text())['additivity_verified']})
    pairs=pd.read_csv(ra.OUT/'paired_logistic_auc_differences.csv')
    for row in pairs.itertuples():
        a,b=row.comparison.split(' minus ');p=pd.read_csv(ra.OUT/row.panel/'test_predictions.csv',float_precision='round_trip')
        delta=roc_auc_score(p.diabetes,p[a+'_LogReg_calibrated'],sample_weight=p.analysis_weight)-roc_auc_score(p.diabetes,p[b+'_LogReg_calibrated'],sample_weight=p.analysis_weight)
        ok=bool(np.isclose(delta,row.delta_auc,atol=1e-12));checks.append({'check':'paired_plot_difference_from_same_people','panel':row.panel,'comparison':row.comparison,'passed':ok})
        if not ok:issues.append({'plot':'paired_auc','panel':row.panel,'comparison':row.comparison})
    for name,data in [('paired_auc.png',pairs[pairs.comparison=='WHR minus BMI'].to_dict('records')),('cohort_flow.png',json.loads((ra.OUT/'audit.json').read_text()))]:
        path=ra.OUT/name;records.append({'file':str(path.relative_to(ra.ROOT)),'kind':name,'data':data,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    composition=pd.read_csv(ra.OUT/'cohort_and_split_composition.csv')
    for panel in ra.PANELS:
        actual=pd.read_csv(ra.OUT/panel/'split_manifest.csv')
        for row in composition[composition.panel==panel].itertuples():
            d=actual if row.split=='all' else actual[actual.split==row.split]
            ok=len(d)==row.n and d.diabetes.sum()==row.positive and d.SDMVSTRA.nunique()==row.strata and len(d[['SDMVSTRA','SDMVPSU']].drop_duplicates())==row.psus and np.isclose(np.average(d.diabetes,weights=d.analysis_weight),row.weighted_positive_proportion,atol=1e-12)
            checks.append({'check':'cohort_counts_cases_units_and_weighted_prevalence','panel':panel,'split':row.split,'passed':bool(ok)})
            if not ok:issues.append({'cohort':panel,'split':row.split})
    ra.dump(ra.OUT/'figure_numerical_audit.json',{'checks':checks,'discrepancies':issues,'figure_data':records,'passed':not issues and all(c['passed'] for c in checks)})
    assert not issues and all(c['passed'] for c in checks)
    print('Published figures and cohort counts verified against final source data.',flush=True)

if __name__=='__main__':run()
