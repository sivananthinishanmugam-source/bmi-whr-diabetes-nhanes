"""Verify resampling endpoints against the complete eight-pattern support.

With three paired strata, every pattern has probability 1/8. The 0.025 and
0.975 endpoints are therefore extrema when all eight patterns are represented
at least that often. This independently checks the exploratory endpoints;
it does not turn them into confidence intervals with nominal coverage.
"""
import itertools
import numpy as np
import pandas as pd
import revision_analysis as ra

def run():
    checks=[]
    for panel in ra.PANELS:
        dest=ra.OUT/panel;frame=pd.read_csv(dest/'test_predictions.csv',float_precision='round_trip')
        strata=[g for _,g in frame.groupby('SDMVSTRA')]
        assert len(strata)==3 and all(g.SDMVPSU.nunique()==2 for g in strata)
        weights=[]
        for chosen in itertools.product(*[sorted(g.SDMVPSU.unique()) for g in strata]):
            mult=np.zeros(len(frame))
            for g,unit in zip(strata,chosen):mult[g.index[g.SDMVPSU==unit]]=2.
            weights.append(frame.analysis_weight.to_numpy()*mult)
        for path in list(dest.glob('*_psu_resampling_intervals.csv'))+list((dest/'neural').glob('*_psu_resampling_intervals.csv')):
            label=path.name.removesuffix('_psu_resampling_intervals.csv')
            if path.parent.name=='neural':
                config,variant=label.rsplit('_',1)
                pr=pd.read_csv(dest/'neural'/f'{config}_test_predictions.csv',float_precision='round_trip').set_index('SEQN').loc[frame.SEQN]
                probability=pr[variant].to_numpy()
            else:probability=frame[label].to_numpy()
            results=pd.DataFrame([ra.metrics(frame.diabetes,probability,w) for w in weights])
            saved=pd.read_csv(path).set_index('metric')
            for metric,row in saved.iterrows():
                a=results[metric].min();b=results[metric].max()
                ok=bool(np.isclose(row.lower_psu,a,atol=2e-6,rtol=1e-6,equal_nan=True) and np.isclose(row.upper_psu,b,atol=2e-6,rtol=1e-6,equal_nan=True))
                checks.append({'source':str(path.relative_to(ra.ROOT)),'metric':metric,'lower_recorded':row.lower_psu,'lower_complete_support':a,'upper_recorded':row.upper_psu,'upper_complete_support':b,'passed':ok})
        pair=pd.read_csv(ra.OUT/'paired_logistic_auc_differences.csv')
        from sklearn.metrics import roc_auc_score
        for row in pair[pair.panel==panel].itertuples():
            a,b=row.comparison.split(' minus ')
            delta=[roc_auc_score(frame.diabetes,frame[a+'_LogReg_calibrated'],sample_weight=w)-roc_auc_score(frame.diabetes,frame[b+'_LogReg_calibrated'],sample_weight=w) for w in weights]
            ok=bool(np.isclose(row.lower_psu,min(delta),atol=1e-12) and np.isclose(row.upper_psu,max(delta),atol=1e-12))
            checks.append({'source':'outputs/paired_logistic_auc_differences.csv','panel':panel,'comparison':row.comparison,'passed':ok})
    ra.dump(ra.OUT/'resampling_endpoint_verification.json',{'unique_patterns':8,'no_nominal_coverage_claim':True,'checks':checks,'passed':all(x['passed'] for x in checks)})
    assert all(x['passed'] for x in checks),'Resampling discrepancy; inspect resampling_endpoint_verification.json'
    print('Verified',len(checks),'resampling endpoint pairs against complete eight-pattern support.',flush=True)

if __name__=='__main__':run()
