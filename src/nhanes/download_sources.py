"""Download missing CDC source files; never overwrite existing raw observations."""
from pathlib import Path
from urllib.request import urlretrieve
from prepare_raw import FILES
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    dest=ROOT.parents[1]/'data/raw';dest.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        path=dest/f'{name}.xpt'
        if path.exists():print('Present',path.name);continue
        url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt'
        urlretrieve(url,path)
        with path.open('rb') as f:assert b'HEADER RECORD' in f.read(100),'Expected SAS XPORT, not an HTML error'
        print('Downloaded',path.name)
