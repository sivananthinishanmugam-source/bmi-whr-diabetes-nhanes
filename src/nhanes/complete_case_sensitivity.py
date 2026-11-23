"""Common-algorithm complete-case sensitivity; not a paired imputation comparison."""
import numpy as np
import pandas as pd
import revision_analysis as ra


def run():
    _,data=ra.load_data();rows=[];predictions=[]
    for panel in ra.PANELS:
        d,nums=ra.get_panel(data,panel)
        manifest=pd.read_csv(ra.OUT/panel/'split_manifest.csv')
        d=d.merge(manifest[['SEQN','split']],on='SEQN',validate='one_to_one')
        # Complete cases for the full BOTH configuration, held common across configurations.
        cc=d.dropna(subset=nums+ra.CATEGORICAL)
        tr=cc[cc.split=='train'];va=cc[cc.split=='validation'];te=cc[cc.split=='test']
        for config,remove in ra.CONFIGS.items():
            base=dict(panel=panel,config=config,cohort_n=len(cc),train_n=len(tr),validation_n=len(va),test_n=len(te),test_cases=int(te.diabetes.sum()))
            if any(x.diabetes.nunique()<2 for x in [tr,va,te]):
                rows.append(dict(**base,status='Not estimable: one class absent'));continue
            cn=[c for c in nums if c not in remove];feats=cn+ra.CATEGORICAL
            pre,model=ra.fit_model('LogReg',tr[feats],tr.diabetes,tr.analysis_weight,cn)
            val,kind=ra.score_model(pre,model,va[feats]);cal=ra.calibrate(val,va.diabetes,va.analysis_weight,kind)
            s,_=ra.score_model(pre,model,te[feats]);p=ra.apply_calibrator(cal,s,kind)
            saved=te[['SEQN','diabetes','analysis_weight']].copy();saved['panel']=panel;saved['config']=config;saved['probability']=p;predictions.append(saved)
            rows.append(dict(**base,status='Exploratory; changed population and potentially informative complete-case selection',**ra.metrics(te.diabetes,p,te.analysis_weight)))
    pd.DataFrame(rows).to_csv(ra.OUT/'complete_case_sensitivity.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(ra.OUT/'complete_case_test_predictions.csv',index=False,float_format='%.17g')
    print(pd.DataFrame(rows)[['panel','config','cohort_n','test_n','test_cases','auc','brier']].to_string(index=False))


if __name__=='__main__':run()
