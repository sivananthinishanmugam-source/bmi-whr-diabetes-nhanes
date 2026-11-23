from pathlib import Path
import json,hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'
if __name__=='__main__':
    assert (OUT/'complete_verification.json').exists()
    audit=json.loads((OUT/'audit.json').read_text())
    comp=pd.read_csv(OUT/'cohort_and_split_composition.csv')
    def cohort(panel):return comp[(comp.panel==panel)&(comp.split=='all')].iloc[0]
    mec=cohort('mec_nonfasting');fast=cohort('fasting_with_glucose')
    def split_label(panel):
        x=comp[comp.panel==panel].set_index('split')
        ns=' / '.join(str(int(x.loc[k,'n'])) for k in ['train','validation','test'])
        return f'New train / validation / test\n{ns} men\n18 / 6 / 6 masked PSUs'
    fig,ax=plt.subplots(figsize=(8.5,8.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor='#f5f6f8',edgecolor='#506880'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.5)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#506880'))
    box(.27,.86,.46,.1,f"Merged records N = {audit['merged_n']:,}\nAdult men N = {audit['adult_men_n']:,}")
    box(.13,.64,.74,.15,f"Primary coding: DIQ010 Yes or No\nN = {audit['definite_outcome_n']:,}; {audit['positive_reports']} positive reports\nExcluded: {audit['borderline_n']} borderline and {audit['unknown_n']} unknown")
    arrow((.5,.86),(.5,.79))
    box(.02,.40,.45,.15,f'Primary nonfasting examination\nN = {mec.n:,}; {mec.positive} positive reports\nPositive WTMEC2YR\nGlucose, LDL and triglycerides omitted')
    box(.53,.40,.45,.15,f'Secondary fasting analysis\nN = {fast.n:,}; {fast.positive} positive reports\nPositive WTSAF2YR; measured glucose\nSame people in glucose-excluded\nsensitivity')
    arrow((.3,.64),(.245,.55));arrow((.7,.64),(.755,.55))
    box(.02,.20,.45,.13,split_label('mec_nonfasting'))
    box(.53,.20,.45,.13,split_label('fasting_with_glucose'))
    arrow((.245,.40),(.245,.33));arrow((.755,.40),(.755,.33))
    ax.text(.5,.11,'Borderline sensitivity: 86 recoded non-positive before eligibility\n82 added in examination panel; 40 added in fasting panels\nSame frozen PSU assignment; positive-report counts unchanged',ha='center',va='center',fontsize=10)
    ax.text(.5,.025,'All 15 strata retain paired PSUs; no PSU crosses partitions',ha='center',fontsize=10)
    fig.savefig(OUT/'cohort_flow.png',dpi=220,bbox_inches='tight');plt.close(fig)
    pairs=pd.read_csv(OUT/'paired_logistic_auc_differences.csv');view=pairs[pairs.comparison=='WHR minus BMI'].copy()
    order=['mec_nonfasting','mec_nonfasting_borderline_nonpositive','fasting_with_glucose','fasting_with_glucose_borderline_nonpositive','fasting_without_glucose','fasting_without_glucose_borderline_nonpositive']
    view=view.set_index('panel').loc[order].reset_index();fig,ax=plt.subplots(figsize=(8,5.5))
    labels=['Primary nonfasting','Primary borderline sensitivity','Secondary with glucose','Secondary borderline sensitivity','Without glucose sensitivity','Without glucose borderline sensitivity']
    for i,row in enumerate(view.itertuples()):
        color='#38698a' if 'borderline' not in row.panel else '#b36b2c'
        ax.hlines(i,row.lower_psu,row.upper_psu,color=color);ax.plot(row.delta_auc,i,'o',color=color);ax.plot([row.lower_psu,row.upper_psu],[i,i],'|',color=color)
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Paired weighted AUC difference WHR minus BMI',title='Exploratory conditional PSU-resampling intervals' )
    fig.tight_layout();fig.savefig(OUT/'paired_auc.png',dpi=220);plt.close(fig)
    (OUT/'code_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
