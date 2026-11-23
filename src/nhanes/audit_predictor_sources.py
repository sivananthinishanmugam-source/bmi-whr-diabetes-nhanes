"""Verify observed cleaned values against candidate source derivations.

Agreement establishes reproducibility of the supplied values, not a scientific
endorsement of all original missing-data decisions or undocumented cutoffs.
"""
import numpy as np
import pandas as pd
import revision_analysis as ra


def run():
    raw,data=ra.load_data();a=data.merge(raw,on='SEQN',suffixes=('_clean','_raw'));rows=[]
    def check(name,source,expected,numeric=True):
        actual=a[name]
        if numeric:
            matches=np.isclose(actual.astype(float),expected.astype(float),equal_nan=True)
        else:
            matches=((actual==expected)|(actual.isna()&expected.isna())).fillna(False)
        rows.append(dict(feature=name,source_or_matching_rule=source,n=len(a),mismatches=int((~matches).sum())))
    for name,col in [('age','RIDAGEYR_raw'),('bmi','BMXBMI'),('heart_rate','BPXPLS'),
                     ('cholesterol_total','LBXTC'),('hdl_cholesterol','LBDHDD'),('ldl_cholesterol','LBDLDL'),
                     ('triglycerides','LBXTR'),('glucose_fasting','LBXGLU'),('poverty_income_ratio','INDFMPIR')]:
        check(name,col,a[col])
    check('waist_to_hip_ratio','BMXWAIST / BMXHIP',a.BMXWAIST_raw/a.BMXHIP_raw)
    for prefix,target in [('BPXSY','systolic_bp'),('BPXDI','diastolic_bp')]:
        x=a[[prefix+str(i) for i in range(1,5)]];v=x.mean(axis=1).where(x.notna().sum(axis=1)>=2)
        rule=f'Mean {prefix}1..4 only if at least two readings'
        if target=='diastolic_bp':
            v=v.where(v>=30);rule+='; means <30 made missing in supplied cleaned data'
        check(target,rule,v)
    check('sleep_hours_per_day','SLD012 weekday/workday sleep hours; 77 and 99 missing',a.SLD012.replace({77:np.nan,99:np.nan}))
    check('alcohol_drinks_per_drinking_day','ALQ130; 777 and 999 missing',a.ALQ130.replace({777:np.nan,999:np.nan}))
    parts=[]
    for flag,day,minute in [('PAQ605','PAQ610','PAD615'),('PAQ620','PAQ625','PAD630'),('PAQ635','PAQ640','PAD645'),('PAQ650','PAQ655','PAD660'),('PAQ665','PAQ670','PAD675')]:
        part=a[day].replace({77:np.nan,99:np.nan})*a[minute].replace({7777:np.nan,9999:np.nan})
        parts.append(part.where(a[flag]!=2,0))
    values=pd.concat(parts,axis=1).sum(axis=1,min_count=1)
    check('physical_activity_minutes_per_week','Sum observed day*minutes in five domains; domain zero for No response; missing if no domain observed; partial sums can underestimate activity',values)
    for name,col in [('ever_smoked_100_cigarettes','SMQ020'),('hypertension_history','BPQ020'),('family_history_diabetes','MCQ300C_raw')]:
        check(name,col+' 1=Yes, 2=No, other=missing',a[col].map({1:'Yes',2:'No'}),False)
    education={1:'Less than 9th grade',2:'9th\u201311th grade',3:'High school graduate/GED',4:'Some college/AA degree',5:'College graduate or above'}
    check('education_level','DMDEDUC2 codes 1..5; special codes and unavailable age groups missing',a.DMDEDUC2.map(education),False)
    ethnicity={1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other/Multiracial'}
    check('ethnicity','RIDRETH3',a.RIDRETH3.map(ethnicity),False)
    result=pd.DataFrame(rows);result.to_csv(ra.OUT/'predictor_source_crosswalk.csv',index=False)
    # Replace the exploratory candidate-check report with the final verified rules.
    result.to_csv(ra.OUT/'numeric_source_crosswalk_checks.csv',index=False)
    print(result[['feature','mismatches']].to_string(index=False))
    assert result.mismatches.sum()==0,'Some source derivations still need investigation.'


if __name__=='__main__':run()
