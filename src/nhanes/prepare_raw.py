"""Rebuild merged NHANES observations directly from the fourteen CDC XPORT files."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = ['DEMO_J','BMX_J','BPX_J','BPQ_J','DIQ_J','GLU_J','HDL_J','TCHOL_J',
         'TRIGLY_J','ALQ_J','PAQ_J','SLQ_J','SMQ_J','MCQ_J']

def run():
    frames=[]; manifest=[]
    for name in FILES:
        path=ROOT.parents[1]/'data/raw'/f'{name}.xpt'
        if not path.exists():
            raise FileNotFoundError(f'{path}: follow DATA_SOURCES.md or run download_sources.py')
        x=pd.read_sas(path,format='xport').replace(5.397605346934028e-79,0.)
        assert not x.SEQN.duplicated().any()
        manifest.append({'file':path.name,'records':len(x),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'download_url':f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt',
                         'documentation_url':f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.htm'})
        if name=='MCQ_J':x=x[['SEQN','MCQ300C']]
        frames.append(x)
    merged=frames[0]
    for frame in frames[1:]:merged=merged.merge(frame,on='SEQN',how='left',validate='one_to_one',sort=False)
    assert len(merged)==9254 and not merged.SEQN.duplicated().any()
    (ROOT.parents[1]/'data/prepared').mkdir(parents=True,exist_ok=True)
    merged.to_csv(ROOT.parents[1]/'data/prepared/NHANES_2017_2018_Merged.csv',index=False)
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/raw_source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Rebuilt {len(merged)} records from {len(FILES)} raw files.',flush=True)

if __name__=='__main__':run()
