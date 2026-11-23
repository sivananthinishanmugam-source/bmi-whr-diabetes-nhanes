"""Single-entry raw-data rerun, reporting and numerical audit for the current package."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,hashlib,importlib.metadata,importlib.util,json,platform,subprocess,sys,time
import revision_analysis as ra
ROOT=Path(__file__).resolve().parent

def run(document_python,analysis_only=False):
    started=time.time();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/'logs').mkdir(exist_ok=True);ra.OUT.mkdir(exist_ok=True)
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    for module in ['revision_analysis','design','neural_revision','check_and_explain','complete_case_sensitivity','audit_predictor_sources','verify_and_report']:
        resolved=Path(importlib.util.find_spec(module).origin).resolve()
        assert resolved==ROOT/(module+'.py'),f'Unexpected imported module: {resolved}'
    ra.dump(ra.OUT/'import_resolution.json',{'local_modules_only':True,'package_root':str(ROOT)})
    ra.dump(ra.OUT/'results_presented_to_author.json',{'presented':False,'reason':'Fresh numerical results require author review before a submission decision'})
    ra.dump(ra.OUT/'pipeline_status.json',{'phase':'running','completed':False,'documents_generated':False,'qa_complete':False})
    def task(script,args=None,executable=None):
        with (ROOT/'logs'/(Path(script).stem+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([executable or sys.executable,'-u',script]+(args or []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',script,flush=True)
    task('prepare_raw.py')
    # A rerun always restores the authoritative fixed split; it does not re-search assignments.
    import shutil
    for name in ['psu_assignment.csv','partition_protocol.json']:shutil.copy2(ROOT/'protocol'/name,ra.OUT/name)
    ra.audit()
    ra.dump(ra.OUT/'analysis_protocol.json',{'primary':'mec_nonfasting','secondary':'fasting_with_glucose','sensitivity':'fasting_without_glucose','outcome_sensitivity':'DIQ010=3 non-positive; all three panels rerun','synthetic_data_included':False,'prior_individual_split_used':False,'psu_partition_frozen':True,'external_validation_performed':False,'interval_label':'exploratory conditional PSU-resampling interval; no nominal coverage asserted','reporting_addition':'trapezoidal PR AUC added to AP during final correction; models unchanged','author_review_required':True,'response_record_purpose':'internal revision record or possible appeal; paper rejected'})
    env={'python':platform.python_version()}
    for package in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost','imbalanced-learn','matplotlib','joblib','tabulate','nbformat','nbclient','ipykernel']:env[package]=importlib.metadata.version(package)
    ra.dump(ra.OUT/'environment.json',env)
    ra.dump(ra.OUT/'neural_environment.json',{p:importlib.metadata.version(p) for p in ['tensorflow','keras','shap']})
    def models(script,args,logname):
        with (ROOT/'logs'/logname).open('w',encoding='utf-8') as log:subprocess.run([sys.executable,'-u',script]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',logname,flush=True)
    with ThreadPoolExecutor(max_workers=1) as classic,ThreadPoolExecutor(max_workers=3) as neural:
        c=classic.submit(models,'revision_analysis.py',['run','--panel','all','--bootstrap','500'],'classical.log')
        ns=[neural.submit(models,'neural_revision.py',['--panel',panel],'neural_'+panel+'.log') for panel in ra.PANELS]
        for future in as_completed(ns):future.result()
        c.result()
    for script in ['audit_predictor_sources.py','complete_case_sensitivity.py','check_and_explain.py','verify_and_report.py','verify_metrics.py','verify_resampling.py','make_figures.py','figure_audit.py']:task(script)
    ra.dump(ra.OUT/'pipeline_status.json',{'phase':'analysis_verified','completed':True,'documents_generated':False,'author_review_required':True})
    if not analysis_only:
        for script in ['create_documents.py','numerical_consistency.py']:task(script,executable=document_python)
        task('create_notebooks.py');task('execute_notebooks.py')
    changed=[p.name for p in ROOT.glob('*.py') if hashlib.sha256(p.read_bytes()).hexdigest()!=code[p.name]]
    assert not changed,'Code changed during frozen final rerun: '+str(changed)
    ra.dump(ra.OUT/'code_sha256.json',code)
    ra.dump(ra.OUT/'clean_run_manifest.json',{'started_utc':stamp,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-started,'entry_point':'run_all.py','analysis_only':analysis_only,'completed_without_manual_intervention':True,'raw_data_rebuilt':True,'all_models_refitted':True,'frozen_assignment_reused':True,'code_unchanged_during_run':True,'analysis_python':sys.executable,'document_python':document_python,'code_sha256':code})
    ra.dump(ra.OUT/'pipeline_status.json',{'phase':'verified','completed':True,'documents_generated':not analysis_only,'numerical_consistency_passed':not analysis_only,'visual_qa_pending':True,'author_review_required':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--document-python',default=sys.executable);parser.add_argument('--analysis-only',action='store_true');args=parser.parse_args()
    try:run(args.document_python,args.analysis_only)
    except Exception as error:
        ra.dump(ra.OUT/'pipeline_status.json',{'phase':'failed','completed':False,'error':str(error)})
        raise
