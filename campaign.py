#!/usr/bin/env python3
"""Single-seed main-paper campaign. No ensemble training or aggregation."""
import os,sys,json,shutil,hashlib,subprocess,time,traceback,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent
PROJECT=Path('/mnt/c/Research/DTI_ablation/dti_stgnn_project/zip_file')
REF=RUNS/'innovation_12h_C2_historical_400ep_pat150_seed77/full/clinical_no_visit_film/seed_77/attempt_01'
BROAD=RUNS/'expanded_1004_clinical_v2_250ep_pat75/full/baseline'
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def status():
 m=load(ROOT/'manifest.json');jobs=m['jobs'];counts={};folds=0
 for j in jobs:
  d=ROOT/'jobs'/j['name'];state=load(d/'status.json')['state'] if (d/'status.json').exists() else 'pending'
  counts[state]=counts.get(state,0)+1
  folds+=sum((d/'run'/f'fold{i}'/'result.json').exists() for i in range(1,6))
  if state in ('preflight','training'):print(j['name'],state,'folds',sum((d/'run'/f'fold{i}'/'result.json').exists() for i in range(1,6)),'/5')
 print(f"Completed {counts.get('complete',0)} / {len(jobs)} new single-model runs; folds {folds} / {5*len(jobs)}; failed {counts.get('failed',0)}")
 print('Budget: queued 250 epochs / patience 75; completed and already-running jobs preserve their recorded budget')
 print('States:',counts,'| existing seed-77 reference reused separately | target: 3 workers x 2 CPU threads (see runtime.json during handover)')
 for j in jobs:
  f=ROOT/'jobs'/j['name']/'status.json'
  if f.exists() and load(f)['state']=='failed':print('FAILED',j['name'],load(f).get('error',''))
 ready={j['name'] for j in jobs if (ROOT/'jobs'/j['name']/'status.json').exists() and load(ROOT/'jobs'/j['name']/'status.json')['state']=='complete'}|{'existing_hacl_seed77'}
 byname={j['name']:j for j in jobs}
 finished_groups=[name for name,arms in m['groups'].items() if set(arms)|{byname[a]['control'] for a in arms if a in byname} <= ready]
 print('Completed categories:',len(finished_groups),'/',len(m['groups']),', '.join(finished_groups) or '(none yet)')
 retry=ROOT/'retry_queue.json'
 if retry.exists():print('Automatic retries:',load(retry).get('state'),'| up to two retries per failed job; partial outputs retained')
 return counts

def command(overrides,dest):
 a=list(load(ROOT/'reference_command.json')['argv']);a[:3]=[sys.executable,'-u',str(ROOT/'source/dti_main_stable_change_v13_history.py')]
 for k,v in dict(seed=77,save_path=str(dest),wandb=False,save_model=False,**overrides).items():
  flags={'--'+k,'--no-'+k};i=3
  while i<len(a):
   if a[i] in flags:
    del a[i]
    if i<len(a) and not str(a[i]).startswith('--'):del a[i]
   else:i+=1
  if v is None:continue
  if isinstance(v,bool):
   if k in {'clinical_only','no_topk_pooling','save_model','wandb','check_only'}:
    if v:a.append('--'+k)
   else:a.append(('--' if v else '--no-')+k)
  else:a.extend(['--'+k,str(v)])
 return a

def prepare():
 if (ROOT/'manifest.json').exists():raise RuntimeError('Manifest already exists; use run/status without overwriting it')
 src=PROJECT/'batch_tools/_ensemble_val_trainer_sources';dst=ROOT/'source';dst.mkdir(exist_ok=True)
 refcmd=load(REF/'runner_command.json');expected=refcmd['model_source_sha256']
 excluded={'run_one_factor_ablations.py','run_dti_ablations.py','make_one_factor_config.py','add_atlas_cli.py','audit_dx_hard_rule.py','add_posttraining_ablations.py','inspect_schaefer17_and_adjacency.py','audit_dti_atlas_roots.py'}
 h=hashlib.sha256()
 for f in sorted(src.glob('*.py')):
  if f.name not in excluded:h.update(f.name.encode());h.update(hashlib.sha256(f.read_bytes()).digest())
 assert h.hexdigest()==expected,'Original source mismatch'
 for f in src.glob('*.py'):shutil.copy2(f,dst/f.name)
 save(ROOT/'reference_command.json',refcmd)
 base=load(REF/'args.json'); jobs=[];seen={};groups={}
 def add(name,changes,group,scope='primary',control='existing_hacl_seed77'):
  changes={k:v for k,v in changes.items() if base.get(k)!=v}
  if not changes:groups.setdefault(group,[]).append('existing_hacl_seed77');return
  key=json.dumps(changes,sort_keys=True)
  if key in seen:groups.setdefault(group,[]).append(seen[key]);return
  seen[key]=name;groups.setdefault(group,[]).append(name)
  jobs.append(dict(name=name,overrides=changes,scope=scope,control=control))
 add('no_change_branch',{'innovation_mode':'none'},'structural')
 cfg=load(PROJECT/'campaign_12h_math_innovation_C2_historical_400ep_pat150_seed77.json')
 for e in cfg['experiments']:
  name=e['name'];o=e['overrides'].copy()
  if name=='reference':o={'visit_film':True};name='visit_film'
  if o.get('clinical_film_mode')=='concordance':o['visit_film']=True
  add(name,o,'structural',control='visit_film' if o.get('clinical_film_mode')=='concordance' else 'existing_hacl_seed77')
 old={e['name']:e['overrides'] for e in load(PROJECT/'dti_one_factor_expanded_1004_clinical_v2_study.json')['experiments']}
 sets={
 'fusion':['fusion_late_add','fusion_late_concat','fusion_late_film','fusion_early','fusion_both'],
 'clinical':['v2_context_only','v2_cognition_summary','v2_cognition_items','v2_clinical_full_items','v2_no_cdr','v2_extended'],
 'modalities':['clinical_only','no_clinical_metadata'],
 'graph_processing':['no_topk_pool','no_dropedge','no_pna','no_feature_clipping','no_feature_standardization'],
 'temporal':['no_prev_dx','no_time_gap','no_temporal_delta','no_temporal_attention'],
 'sampling':['ratio_natural','ratio_1to1','ratio_2to1','ratio_5to1'],
 'crt':['no_crt','crt_10_epochs','crt_50_epochs','crt_balanced_sampling','crt_lws_instead']}
 add('fusion_late_gated',{'metadata_fusion_operator':'gated'},'fusion')
 for group,names in sets.items():
  for n in names:
   o=old[n].copy()
   if n=='clinical_only':o['innovation_mode']='none'
   add(n,o,group)
 for n in ['tract_static_atlas','tract_static_mean','tract_dynamic_topk20']:
  o=dict(old[n],innovation_mode='node');add(n,o,'connectivity',control='node_only')
 # Summary-without-CDR is distinct from primary-items-without-CDR.
 clinical=Path(base['expanded_clinical_root'])/'past126_plus00'
 local=ROOT/'clinical_summary_no_cdr/past126_plus00';local.mkdir(parents=True)
 for f in clinical.iterdir():
  if f.name!='feature_columns.json':(local/f.name).symlink_to(f,target_is_directory=f.is_dir())
 schema=load(clinical/'feature_columns.json');features=schema['ordered_features']
 schema['feature_sets']['C2_noCDR']=[i for i in schema['feature_sets']['C2_summary'] if '__CDR__' not in features[i]]
 save(local/'feature_columns.json',schema)
 phase=load(PROJECT/'dti_one_factor_expanded_1004_clinical_v2_film_phase2.json')
 for e in phase['experiments']:
  o=e['overrides'].copy()
  if e['name']=='film_no_cdr_summary':o['expanded_clinical_root']=str(local.parent)
  add(e['name'],o,'film_followup')
 for n in [2,4,5]:add(f'primary_depth{n}',{'gnn_num_layers':n},'depth')
 b=load(BROAD/'args.json')
 for n in [2,3,4,5]:
  o={k:b[k] for k in ['dti_root','selected_csv','manifest_csv','expanded_clinical_root']};o['gnn_num_layers']=n
  add(f'broader_depth{n}',o,'depth','broader','broader_depth3')
 manifest={'seed':77,'split_seed':77,'folds':5,'workers':3,'threads_per_worker':2,'source_hash':expected,'reference':str(REF),'excluded':['ensemble aggregation','ensemble retraining','independent-seed ensemble members','supplement-only ablations'],'groups':groups,'jobs':jobs,'notes':['All arms inherit current HACL settings, 400 epochs maximum and patience 150.','Support variants use node-only change and compare to node_only because edge-change requires fixed support.','Broader-cohort depth sensitivity deliberately changes cohort; not a one-factor primary-cohort comparison.','Clinical concordance requires visit FiLM and compares against visit_film.','Identical configurations appearing in multiple figures are trained only once.','Test folds were previously inspected; these remain retrospective ablations.']}
 save(ROOT/'manifest.json',manifest)
 for j in jobs:save(ROOT/'jobs'/j['name']/'command.json',{'argv':command(j['overrides'],ROOT/'jobs'/j['name']/'run')})
 save(ROOT/'runtime.json',{'state':'prepared','new_runs':len(jobs),'new_folds':len(jobs)*5})
 print('Prepared',len(jobs),'new single-seed runs;',len(jobs)*5,'fold fits; no ensembles.')

def audit_windows(run,reference,full=False):
 import pandas as pd
 a=pd.read_csv(reference/'all_windows_distribution.csv');b=pd.read_csv(run/'all_windows_distribution.csv')
 keys=['subject_id','input_dates','target_date','stable_change_label']
 keys=[k for k in keys if k in a and k in b]
 assert len(keys)>=3
 pd.testing.assert_frame_equal(a[keys].sort_values(keys).reset_index(drop=True),b[keys].sort_values(keys).reset_index(drop=True),check_dtype=False)
 if full:
  for fold in range(1,6):
   for part in ['val','test']:
    a=pd.read_csv(reference/f'fold{fold}'/f'{part}_windows.csv');b=pd.read_csv(run/f'fold{fold}'/f'{part}_windows.csv')
    cols=[k for k in keys if k in a and k in b]
    pd.testing.assert_frame_equal(a[cols].sort_values(cols).reset_index(drop=True),b[cols].sort_values(cols).reset_index(drop=True),check_dtype=False)

def execute(j):
 d=ROOT/'jobs'/j['name'];out=d/'run';sf=d/'status.json'
 if sf.exists() and load(sf)['state']=='complete':return
 try:
  if out.exists():raise RuntimeError('Partial output retained; inspect before retrying, no overwrite permitted')
  reference=BROAD if j['scope']=='broader' else REF
  save(sf,{'state':'preflight','started':time.time()})
  check=command(dict(j['overrides'],check_only=True),d/'preflight')
  with (d/'preflight.log').open('w') as log:subprocess.run(check,cwd=ROOT/'source',env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
  audit_windows(d/'preflight',reference)
  with (d/'training.log').open('w') as log:
   proc=subprocess.Popen(command(j['overrides'],out),cwd=ROOT/'source',env=ENV,stdout=log,stderr=subprocess.STDOUT)
   save(sf,{'state':'training','pid':proc.pid,'started':time.time()});rc=proc.wait()
  if rc:raise RuntimeError(f'Trainer exit {rc}; see training.log')
  assert (out/'cv_summary.csv').exists(),'No summary'
  assert '[cRT] first attempt failed' not in (d/'training.log').read_text(),'cRT fallback changes protocol'
  audit_windows(out,reference,True)
  save(sf,{'state':'complete','finished':time.time(),'folds':5,'audit':'cohort and validation/test windows match reference'})
 except Exception as e:
  save(sf,{'state':'failed','error':str(e),'finished':time.time()});(d/'failure.txt').write_text(traceback.format_exc())
 print(j['name'],load(sf)['state'],flush=True)

def summarize():
 import pandas as pd
 rows=[]
 for j in load(ROOT/'manifest.json')['jobs']:
  d=ROOT/'jobs'/j['name'];s=d/'status.json'
  if s.exists() and load(s)['state']=='complete':
   df=pd.read_csv(d/'run/cv_summary.csv').set_index('metric')
   row={'name':j['name'],'scope':j['scope'],'control':j['control']}
   for k in ['balanced_acc','auc']:
    row[k+'_mean']=float(df.loc[k,'mean']);row[k+'_std']=float(df.loc[k,'std'])
   rows.append(row)
 pd.DataFrame(rows).to_csv(ROOT/'completed_metrics.csv',index=False)

def run():
 lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert len(os.sched_getaffinity(0))>=6,'Six CPU cores required'
 probe=subprocess.run([sys.executable,'-c','import torch; assert torch.get_num_threads()==2; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'],env=ENV,capture_output=True,text=True,check=True)
 m=load(ROOT/'manifest.json');save(ROOT/'runtime.json',{'state':'running','pid':os.getpid(),'workers':3,'threads_per_worker':2,'gpu':probe.stdout.strip(),'started':time.time()})
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(execute,j) for j in m['jobs']]
  for f in as_completed(futures):f.result();summarize()
 counts=status();save(ROOT/'runtime.json',{'state':'complete' if counts.get('failed',0)==0 else 'finished_with_failures','finished':time.time(),'counts':counts})

def retry_failed():
 import importlib
 # Wait without changing the running scheduler or any child trainer.
 lock=(ROOT/'campaign.lock').open('a')
 save(ROOT/'retry_queue.json',{'state':'queued','policy':'all failed jobs; at most two automatic retry attempts per job','additional_control':'hacl_control_250','workers':3,'threads_per_worker':2})
 print('Waiting for active campaign; all failed jobs will be retried automatically.',flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 history_path=ROOT/'retry_history.json'
 history=load(history_path) if history_path.exists() else {}
 for round_id in range(3):
  lock=(ROOT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   manifest=load(ROOT/'manifest.json');pending=[]
   # Run the matched control before retries, then remaining unfinished jobs.
   jobs=sorted(manifest['jobs'],key=lambda j:j['name']!='hacl_control_250')
   for j in jobs:
    d=ROOT/'jobs'/j['name'];sf=d/'status.json'
    if not sf.exists():pending.append(j);continue
    state=load(sf)
    if state['state']=='complete':continue
    if state['state']!='failed':raise RuntimeError('Unexpected active state after scheduler released lock: '+j['name'])
    attempts=history.setdefault(j['name'],[])
    if len(attempts)>=2:continue
    archive=ROOT/'archived_retries'/j['name']/('attempt_'+str(len(attempts)+1))
    archive.parent.mkdir(parents=True,exist_ok=True)
    assert ROOT.resolve() in d.resolve().parents and ROOT.resolve() in archive.resolve().parents
    assert not archive.exists()
    shutil.move(str(d),str(archive));d.mkdir()
    record={'attempt':len(attempts)+1,'reason':state.get('error'),'preserved_output':str(archive),'started':time.time(),'training_budget':j['training_budget']}
    attempts.append(record);save(history_path,history)
    save(d/'retry_provenance.json',record)
    save(d/'command.json',{'argv':command(j['overrides'],d/'run'),'training_budget':j['training_budget']})
    pending.append(j)
   if not pending:break
   save(ROOT/'retry_queue.json',{'state':'running','round':round_id+1,'jobs':[j['name'] for j in pending],'retry_limit_per_job':2})
   save(ROOT/'runtime.json',{'state':'running_retries','pid':os.getpid(),'workers':3,'threads_per_worker':2,'allocated_cpu_threads':6})
   with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(execute,j) for j in pending]
    for future in as_completed(futures):future.result();summarize()
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 counts=status();save(ROOT/'retry_queue.json',{'state':'finished' if not counts.get('failed') else 'retry_limit_reached','counts':counts,'retry_limit_per_job':2})
 save(ROOT/'runtime.json',{'state':'complete' if not counts.get('failed') else 'finished_with_failures','workers':3,'threads_per_worker':2,'counts':counts})

def retry_interrupted():
 return retry_failed()
if __name__=='__main__':
 action=sys.argv[1] if len(sys.argv)>1 else 'status'
 {'prepare':prepare,'run':run,'status':status,'summarize':summarize,'retry_interrupted':retry_interrupted,'retry_failed':retry_failed}[action]()

