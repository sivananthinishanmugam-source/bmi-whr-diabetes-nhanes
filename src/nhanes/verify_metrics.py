"""Recompute every saved performance metric from final predictions, not manuscript text."""
import json
import numpy as np
import pandas as pd
import revision_analysis as ra

def run():
    checks=[]
    def check(source,row,calculated):
        for key,expected in calculated.items():
            actual=getattr(row,key,None)
            if isinstance(expected,bool):ok=bool(actual)==expected
            elif expected is None:ok=pd.isna(actual)
            else:ok=bool(np.isclose(actual,expected,atol=1e-9,rtol=1e-8,equal_nan=True))
            checks.append({'source':source,'config':row.config,'model':getattr(row,'model','LogReg'),'variant':getattr(row,'variant','calibrated'),'metric':key,'recorded':actual,'recalculated':expected,'passed':ok})
    for panel in ra.PANELS:
        dest=ra.OUT/panel
        pred=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip')
        for row in pd.read_csv(dest/'test_results.csv').itertuples():
            key=f'{row.config}_{row.model}_{row.variant}'
            check(str((dest/'test_results.csv').relative_to(ra.ROOT)),row,ra.metrics(pred.diabetes,pred[key],pred.analysis_weight))
        for row in pd.read_csv(dest/'neural/test_results.csv').itertuples():
            pr=pd.read_csv(dest/'neural'/f'{row.config}_test_predictions.csv',float_precision='round_trip')
            check(str((dest/'neural/test_results.csv').relative_to(ra.ROOT)),row,ra.metrics(pr.outcome,pr[row.variant],pr.weight))
        for path in dest.glob('*_threshold_sensitivity.csv'):
            label=path.name.removesuffix('_threshold_sensitivity.csv')
            for row in pd.read_csv(path).itertuples():
                for metric,expected in ra.metrics(pred.diabetes,pred[label+'_calibrated'],pred.analysis_weight,row.threshold,calibration=False).items():
                    checks.append({'source':str(path.relative_to(ra.ROOT)),'metric':metric,'threshold':row.threshold,'recorded':getattr(row,metric),'recalculated':expected,'passed':bool(np.isclose(getattr(row,metric),expected,atol=1e-9,rtol=1e-8,equal_nan=True))})
    cc=pd.read_csv(ra.OUT/'complete_case_test_predictions.csv',float_precision='round_trip')
    for row in pd.read_csv(ra.OUT/'complete_case_sensitivity.csv').itertuples():
        pr=cc[(cc.panel==row.panel)&(cc.config==row.config)]
        check('outputs/complete_case_sensitivity.csv',row,ra.metrics(pr.diabetes,pr.probability,pr.analysis_weight))
    for row in pd.read_csv(ra.OUT/'borderline_models_on_definite_test.csv').itertuples():
        original=pd.read_csv(ra.OUT/row.panel/'test_predictions.csv',float_precision='round_trip')
        added=pd.read_csv(ra.OUT/(row.panel+'_borderline_nonpositive')/'test_predictions.csv',float_precision='round_trip').set_index('SEQN').loc[original.SEQN]
        check('outputs/borderline_models_on_definite_test.csv',row,ra.metrics(original.diabetes,added[row.config+'_LogReg_calibrated'],original.analysis_weight))
    ra.dump(ra.OUT/'metric_recalculation.json',{'checks':checks,'count':len(checks),'mismatches':sum(not c['passed'] for c in checks),'passed':all(c['passed'] for c in checks)})
    assert all(c['passed'] for c in checks),'Metric discrepancy; see metric_recalculation.json'
    print('Recomputed',len(checks),'performance and threshold metric values, zero discrepancies.',flush=True)

if __name__=='__main__':run()
