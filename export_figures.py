"""Export completed main-paper ablation groups without modifying the manuscript."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,r'C:\Users\brett\OneDrive\Documents\ChatGPT\AgenticAD\paper_tools\python_packages')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.bbox':'tight'})
manifest=json.loads((ROOT/'manifest.json').read_text())
reference=Path('C:/'+manifest['reference'].split('/mnt/c/',1)[1])
OUT=ROOT/'main_paper_figures';OUT.mkdir(exist_ok=True)
names={'main_branches':'main_branches','structural':'innovation_campaign','fusion':'fusion','clinical':'clinical','modalities':'modalities','connectivity':'connectivity','graph_processing':'graph_processing','temporal':'temporal_controls','sampling':'sampling','crt':'crt','film_followup':'film_phase2','depth':'depth'}
labels={'no_change_branch_250':'Without explicit change encoder (250/75)','existing_hacl_seed77':'HACL single-model control','no_change_branch':'No ROI/tract change branch','visit_film':'Visit + late FiLM','node_only':'ROI change only','edge_only':'Tract change only','no_clinical_metadata':'No clinical branch','v2_context_only':'Context only','v2_cognition_summary':'Cognition summary','v2_cognition_items':'Cognition items','v2_clinical_full_items':'Primary clinical items','v2_no_cdr':'Primary items without CDR','v2_extended':'Extended clinical','film_no_cdr_summary':'Clinical summary without CDR','hacl_control_250':'HACL control (250/75)','clinical_no_visit_film':'HACL single-model control'}

def summary(name):
 d=reference if name=='existing_hacl_seed77' else ROOT/'jobs'/name/'run'
 s=pd.read_csv(d/'cv_summary.csv').set_index('metric')
 return {k:float(s.loc[metric,col]) for k,metric,col in [('BA','balanced_acc','mean'),('BA_SD','balanced_acc','std'),('AUC','auc','mean'),('AUC_SD','auc','std')]}

def update():
 complete=[];failed=[]
 for j in manifest['jobs']:
  p=ROOT/'jobs'/j['name']/'status.json'
  if p.exists():
   state=json.loads(p.read_text())['state']
   if state=='complete':complete.append(j['name'])
   elif state=='failed':failed.append(j['name'])
 ready=set(complete)|{'existing_hacl_seed77'};done=[]
 byname={j['name']:j for j in manifest['jobs']}
 for group,arms in manifest['groups'].items():
  if not set(arms)<=ready:continue
  controls=list(dict.fromkeys(byname[a]['control'] for a in arms if a in byname))
  if 'existing_hacl_seed77' in arms and 'existing_hacl_seed77' not in controls:controls.insert(0,'existing_hacl_seed77')
  if not set(controls)<=ready:continue
  arms=list(dict.fromkeys(controls+arms));file=names[group]
  if (OUT/(file+'.pdf')).exists():done.append(file);continue
  rows=[dict(name=n,control=byname.get(n,{}).get('control',n),n_epochs=byname.get(n,{}).get('training_budget',{}).get('n_epochs',400),patience=byname.get(n,{}).get('training_budget',{}).get('patience',150),**summary(n)) for n in arms];df=pd.DataFrame(rows)
  df.to_csv(OUT/(file+'.csv'),index=False)
  fig,axs=plt.subplots(1,2,figsize=(10,max(3,.35*len(rows)+1.6)),sharey=True)
  for ax,metric in zip(axs,['BA','AUC']):
   ax.errorbar(df[metric],range(len(df)),xerr=df[metric+'_SD'],fmt='o',color='#254d72',ecolor='#8294a3',capsize=2)
   for ci,control in enumerate(controls):
    ref=summary(control)[metric];ax.axvline(ref,color=['#a64725','#376c96','#765b8d','#596b42'][ci%4],ls='--',label=labels.get(control,control.replace('_',' '))+f' {ref:.4f}')
   ax.set_yticks(range(len(df)),[labels.get(row['name'],row['name'].replace('_',' ').capitalize())+f" [{row['n_epochs']}/{row['patience']}]" for row in rows]);ax.set_xlabel('Balanced accuracy' if metric=='BA' else 'ROC-AUC')
   ax.grid(axis='x',alpha=.2);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=9)
  axs[0].invert_yaxis()
  fig.suptitle(group.replace('_',' ').title()+' | current HACL architecture',y=1.03,fontsize=13)
  note='Labels: maximum epochs / patience; compare each arm with its budget-matched control'
  if group=='depth':note='Primary and broader cohorts shown separately; seed 77; five test folds'
  if group=='connectivity':note='Matched ROI-change-only control; tract-change branch excluded in all support arms'
  fig.text(.5,-.025,note,ha='center',fontsize=9)
  fig.tight_layout();fig.savefig(OUT/(file+'.pdf'));fig.savefig(OUT/(file+'.png'),dpi=180);plt.close(fig);done.append(file)
 state={'completed_runs':len(complete),'total_runs':len(manifest['jobs']),'failed_runs':failed,'exported_figure_groups':done,'total_figure_groups':len(names),'existing_ensemble_figures':'unchanged; not retrained or regenerated','manuscript':'existing paper and supplement untouched; exports await results review'}
 (OUT/'status.json').write_text(json.dumps(state,indent=2))
 if set(manifest.get('priority_jobs',[]))<=ready and manifest.get('priority_jobs'):
  with (ROOT/'branch_paper_refresh.log').open('a') as log:subprocess.run([sys.executable,'-X','utf8',r'C:\Research\DTI_ablation\paper_revision_20261116\refresh_branch_paper.py'],stdout=log,stderr=subprocess.STDOUT)
 retry=json.loads((ROOT/'retry_queue.json').read_text()) if (ROOT/'retry_queue.json').exists() else {}
 return len(complete)+len(failed)==len(manifest['jobs']) and retry.get('state') in ('finished','retry_limit_reached')

if __name__=='__main__':
 while True:
  finished=update()
  if finished or '--once' in sys.argv:break
  time.sleep(30)
