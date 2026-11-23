"""Reconstruct adult male predictors and freeze design-unit partitions before fitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
SOURCE = ROOT.parents[1] / 'data'
ZERO = 5.397605346934028e-79
SPLIT_SEED = 20261012
NUMERIC_SOURCE = {'age':'RIDAGEYR','bmi':'BMXBMI','heart_rate':'BPXPLS',
    'cholesterol_total':'LBXTC','hdl_cholesterol':'LBDHDD','ldl_cholesterol':'LBDLDL',
    'triglycerides':'LBXTR','glucose_fasting':'LBXGLU','poverty_income_ratio':'INDFMPIR'}

def reconstruct():
    raw = pd.read_csv(SOURCE/'prepared/NHANES_2017_2018_Merged.csv',float_precision='round_trip').replace(ZERO,0.)
    a = raw[(raw.RIAGENDR==1)&(raw.RIDAGEYR>=18)].copy()
    d = a[['SEQN','SDMVSTRA','SDMVPSU','WTMEC2YR','DIQ010','RIAGENDR','RIDAGEYR','RIDSTATR','BMXWAIST','BMXHIP','MCQ300C']].copy()
    for name,col in NUMERIC_SOURCE.items():d[name]=a[col]
    assert np.allclose(a.WTSAF2YR_x,a.WTSAF2YR_y,equal_nan=True)
    d['WTSAF2YR']=a.WTSAF2YR_x
    d['waist_to_hip_ratio']=a.BMXWAIST/a.BMXHIP
    for prefix,target in [('BPXSY','systolic_bp'),('BPXDI','diastolic_bp')]:
        x=a[[prefix+str(i) for i in range(1,5)]]
        d[target]=x.mean(axis=1).where(x.notna().sum(axis=1)>=2)
        if target=='diastolic_bp':d[target]=d[target].where(d[target]>=30)
    d['sleep_hours_per_day']=a.SLD012.replace({77:np.nan,99:np.nan})
    d['alcohol_drinks_per_drinking_day']=a.ALQ130.replace({777:np.nan,999:np.nan})
    parts=[]
    for flag,day,minute in [('PAQ605','PAQ610','PAD615'),('PAQ620','PAQ625','PAD630'),('PAQ635','PAQ640','PAD645'),('PAQ650','PAQ655','PAD660'),('PAQ665','PAQ670','PAD675')]:
        part=a[day].replace({77:np.nan,99:np.nan})*a[minute].replace({7777:np.nan,9999:np.nan})
        parts.append(part.where(a[flag]!=2,0))
    d['physical_activity_minutes_per_week']=pd.concat(parts,axis=1).sum(axis=1,min_count=1)
    for name,col in [('ever_smoked_100_cigarettes','SMQ020'),('hypertension_history','BPQ020'),('family_history_diabetes','MCQ300C')]:
        d[name]=a[col].map({1:'Yes',2:'No'})
    d['education_level']=a.DMDEDUC2.map({1:'Less than 9th grade',2:'9th–11th grade',3:'High school graduate/GED',4:'Some college/AA degree',5:'College graduate or above'})
    d['ethnicity']=a.RIDRETH3.map({1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other/Multiracial'})
    d['diabetes']=(d.DIQ010==1).astype(int)
    d['psu_group']=d.SDMVSTRA.astype(int).astype(str)+'_'+d.SDMVPSU.astype(int).astype(str)
    return raw,d.reset_index(drop=True)

def verify_reconstruction():
    _,d=reconstruct()
    if not (SOURCE/'reference/definite_predictor_benchmark.csv').exists():
        (OUT/'reconstruction_verification.json').write_text(json.dumps({'optional_historical_benchmark':'not supplied; raw reconstruction remains complete'}))
        return
    c=pd.read_csv(SOURCE/'reference/definite_predictor_benchmark.csv',float_precision='round_trip').replace(ZERO,0.)
    a=d[d.DIQ010.isin([1,2])].set_index('SEQN',drop=False).loc[c.SEQN]
    checks=[]
    for col in c.columns:
        x=a[col].to_numpy();y=c[col].to_numpy()
        if pd.api.types.is_numeric_dtype(c[col]):match=np.isclose(x.astype(float),y.astype(float),equal_nan=True)
        else:match=(x==y)|(pd.isna(x)&pd.isna(y))
        checks.append({'column':col,'mismatches':int((~match).sum())})
    assert sum(x['mismatches'] for x in checks)==0,checks
    (OUT/'reconstruction_verification.json').write_text(json.dumps(checks,indent=2))

def freeze_splits():
    path=OUT/'psu_assignment.csv'
    if path.exists():return pd.read_csv(path)
    _,d=reconstruct();d=d[d.DIQ010.isin([1,2])]
    strata=sorted(d.SDMVSTRA.unique())
    assert len(strata)==15 and (d.groupby('SDMVSTRA').SDMVPSU.nunique()==2).all()
    # Both masked PSUs stay together. Counts and labels are used for stratification,
    # never predictors, fitted models, calibration, or performance estimates.
    columns=[]
    for wt in ['WTMEC2YR','WTSAF2YR']:
        eligible=d[d[wt]>0]
        for name,values in [('n',np.ones(len(eligible))),('positive',(eligible.diabetes==1).astype(float)),('negative',(eligible.diabetes==0).astype(float)),('weight',eligible[wt])]:
            t=eligible.assign(value=values).groupby('SDMVSTRA').value.sum().reindex(strata,fill_value=0)
            columns.append(t.to_numpy()/t.sum())
    balance=np.array(columns).T
    rng=np.random.default_rng(SPLIT_SEED);best=None
    for i in range(10000):
        order=rng.permutation(len(strata));chunks=[order[:9],order[9:12],order[12:]]
        shares=np.array([balance[x].sum(axis=0) for x in chunks]);loss=float(np.square(shares-np.array([.6,.2,.2])[:,None]).sum())
        if best is None or loss<best[0]:best=(loss,chunks,i)
    mapping={strata[ix]:split for split,chunk in zip(['train','validation','test'],best[1]) for ix in chunk}
    assignment=d[['SDMVSTRA','SDMVPSU','psu_group']].drop_duplicates().sort_values(['SDMVSTRA','SDMVPSU'])
    assignment['split']=assignment.SDMVSTRA.map(mapping);assignment.to_csv(path,index=False)
    metadata={'partition_seed':SPLIT_SEED,'random_candidate_assignments':10000,'chosen_candidate_index':best[2],
      'target_strata_counts':{'train':9,'validation':3,'test':3},'target_person_fractions':[.6,.2,.2],
      'assignment_unit':'paired masked PSUs within an intact SDMVSTRA','psu_identity':'SDMVSTRA plus SDMVPSU',
      'balance_features':'primary definite-outcome eligible MEC and fasting counts, positive/negative counts, and weight totals',
      'balance_loss':best[0],'model_performance_used':False,'frozen_before_model_fitting':True,
      'limitations':'Only 15 public masked strata; partitions contain different strata. Test intervals condition on held-out strata and fitted models; not full-survey population uncertainty. Prior study used these participants.'}
    (OUT/'partition_protocol.json').write_text(json.dumps(metadata,indent=2))
    return assignment

def load(panel):
    raw,d=reconstruct()
    include_borderline='borderline_nonpositive' in panel
    d=d[d.DIQ010.isin([1,2,3] if include_borderline else [1,2])].copy()
    if panel.startswith('fasting'):
        d=d[(d.WTSAF2YR>0)&d.glucose_fasting.notna()].copy();d['analysis_weight']=d.WTSAF2YR
    else:d=d[d.WTMEC2YR>0].copy();d['analysis_weight']=d.WTMEC2YR
    assignment=freeze_splits();d=d.merge(assignment,on=['SDMVSTRA','SDMVPSU','psu_group'],validate='many_to_one')
    assert d.split.notna().all()
    groups=[set(g.psu_group) for _,g in d.groupby('split')]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert all(g.diabetes.nunique()==2 for _,g in d.groupby('split'))
    return d.reset_index(drop=True)

def audit(panels):
    OUT.mkdir(exist_ok=True);verify_reconstruction();freeze_splits()
    raw,d=reconstruct();rows=[]
    for panel in panels:
        x=load(panel)
        for split,g in [('all',x)]+list(x.groupby('split')):
            rows.append(dict(panel=panel,split=split,n=len(g),positive=int(g.diabetes.sum()),nonpositive=int((g.diabetes==0).sum()),borderline=int((g.DIQ010==3).sum()),psus=g.psu_group.nunique(),strata=g.SDMVSTRA.nunique(),weight_total=g.analysis_weight.sum(),weighted_positive_proportion=np.average(g.diabetes,weights=g.analysis_weight)))
        x.to_csv(OUT/f'{panel}_analysis_data.csv',index=False)
        dest=OUT/panel;dest.mkdir(exist_ok=True)
        x[['SEQN','DIQ010','diabetes','analysis_weight','SDMVSTRA','SDMVPSU','psu_group','split']].to_csv(dest/'split_manifest.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'cohort_and_split_composition.csv',index=False)
    facts={'merged_n':len(raw),'adult_men_n':len(d),'definite_outcome_n':int(d.DIQ010.isin([1,2]).sum()),'positive_reports':int((d.DIQ010==1).sum()),'borderline_n':int((d.DIQ010==3).sum()),'unknown_n':int((d.DIQ010==9).sum()),'primary':'mec_nonfasting','secondary':'fasting_with_glucose','sensitivity':'fasting_without_glucose','borderline_sensitivity':'DIQ010=3 grouped with non-positive; not an assertion of normoglycemia','outcome':'cross-sectional self-reported diagnosed diabetes; type 1 and type 2 not distinguished','synthetic_dataset':{'excluded_from_all_analysis':True,'contains_real_patient_records':False,'provenance_basis':'user correction based on uploader description','url':'https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset'},'external_validation_performed':False,'hba1c_used':False}
    (OUT/'audit.json').write_text(json.dumps(facts,indent=2))
    sources={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.suffix in ['.csv','.xpt']}
    (OUT/'source_sha256.json').write_text(json.dumps(sources,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return facts
