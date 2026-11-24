#!/usr/bin/env python3
"""Attach v2 ADNI clinical data to the EXISTING DTI model interface, fold-safely.

The clinical preprocessing is fitted per train subject and never reads outcome
columns. Applies only to graph visits present in the v2 visit CSV; visit graphs
which are targets-only receive missing clinical tensors. The input DTI graph x,
edge_index and labels are never modified. No fMRI is used.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch

from clinical_fold_preprocessing_v2 import ClinicalFoldPreprocessor, assert_disjoint_folds


def _normalized_date(x):
    dt = pd.to_datetime(x, errors='coerce')
    if pd.isna(dt):
        raise ValueError(f'Invalid DTI visit date: {x!r}')
    return dt.normalize().strftime('%Y-%m-%d')


def _canonical_sid(x):
    import re
    return re.sub(r'__seg\d+$', '', str(x).strip())


def _v2_folder(root, alignment):
    if alignment not in {'past126_plus00', 'past126_plus07', 'past126_plus15'}:
        raise ValueError('Alignment must be past126_plus00, past126_plus07 or past126_plus15')
    directory = Path(root)/alignment
    expected = ('visit_features_raw.csv','feature_columns.json',
                'window_outcomes_DO_NOT_USE_AS_FEATURES.csv','clinical_sequences.jsonl')
    missing = [s for s in expected if not (directory/s).is_file()]
    if missing:
        raise FileNotFoundError(f'Clinical alignment folder missing {missing}: {directory}')
    return directory


def assert_v2_cohort(root, alignment, training_window_frame):
    """Ensure actual trainer windows equal v2 source windows (subject/date/outcome)."""
    directory = _v2_folder(root, alignment)
    o = pd.read_csv(directory/'window_outcomes_DO_NOT_USE_AS_FEATURES.csv', low_memory=False)
    cols = ('subject_id','last_input_date','target_date','stable_change_label')
    if not set(cols).issubset(o):
        raise ValueError('v2 outcomes file missing expected matching columns')
    x = training_window_frame.copy()
    if not {'subject_id','input_dates','target_date','stable_change_label'}.issubset(x):
        raise ValueError('trainer window_frame() is missing expected columns')
    x['last_input_date'] = x['input_dates'].astype(str).map(lambda s: s.split(';')[-1])
    def keys(frame):
        return Counter(( _canonical_sid(r.subject_id), _normalized_date(r.last_input_date),
                         _normalized_date(r.target_date), int(r.stable_change_label))
                       for r in frame[list(cols)].itertuples(index=False))
    a,b = keys(x),keys(o)
    if a != b:
        only_trainer = sum((a-b).values()); only_v2 = sum((b-a).values())
        raise ValueError(f'TRAINER/V2 COHORT MISMATCH: trainer-only {only_trainer} windows; '
                         f'v2-only {only_v2} windows. Refusing clinical training.')
    return len(x)


def apply_v2_clinical_fold(dataset, split, fold_dir, root, alignment='past126_plus00',
                           feature_set='C2_summary'):
    """Inject per-visit, train-standardized tensors into DTI dataset for one fold.

    IMPORTANT: invoke AFTER getting train/val/test subject splits but BEFORE
    constructing the fold's temporal classifier and all fold data loaders.
    Do NOT use together with a pretrained metadata imputer whose dimension is
    tied to the legacy ADNIMERGE numeric representation.
    """
    directory=_v2_folder(root, alignment)
    train_s={_canonical_sid(x) for x in split['train_subjects']}
    val_s={_canonical_sid(x) for x in split['val_subjects']}
    test_s={_canonical_sid(x) for x in split['test_subjects']}
    assert_disjoint_folds(train_s,val_s,test_s)
    pp=ClinicalFoldPreprocessor.fit(
        directory/'visit_features_raw.csv',
        directory/'feature_columns.json',
        train_s,feature_set=feature_set)
    # The table is indexed by input DTI scan; outcome columns never loaded.
    visits=pd.read_csv(directory/'visit_features_raw.csv',low_memory=False)
    visits['subject_id']=visits.subject_id.astype(str).map(_canonical_sid)
    visits['scan_date']=pd.to_datetime(visits.scan_date,errors='raise').dt.strftime('%Y-%m-%d')
    if visits.duplicated(['subject_id','scan_date']).any():
        raise ValueError('Duplicate subject/scan in v2 raw visit table')
    transform=pp.transform_dataframe(visits)
    keys={(r.subject_id,r.scan_date):i for i,r in enumerate(visits[['subject_id','scan_date']].itertuples(index=False))}
    num_dim=len(pp.numeric_fields);cat_dim=len(pp.categorical_fields)
    # Ensure model constructor reads new dimensions by updating metadata schema.
    mt=dataset.meta_table
    mt.num_fields=list(pp.numeric_fields)
    mt.cat_fields=list(pp.categorical_fields)
    # dataset.meta_cat_cardinalities computes len(mapping)+1. Values are in 1..K.
    mt.cat_maps={k:{x:i for i,x in enumerate(sorted(v, key=lambda q:float(q)))}
                 for k,v in pp.categorical_vocab.items()}
    first_seen={}
    matched=0;unmatched=0;missing_by_cohort=Counter()
    all_graphs=sorted(dataset.data_list,key=lambda d:(str(d.subject_id),_normalized_date(d.visit_date)))
    for graph in all_graphs:
        sid=_canonical_sid(graph.subject_id)
        scan=_normalized_date(graph.visit_date)
        j=keys.get((sid,scan))
        if j is None:
            num=np.zeros(num_dim,dtype=np.float32)
            mask=np.zeros(num_dim,dtype=np.float32)
            cat=np.zeros(cat_dim,dtype=np.int64)
            cat_mask=np.zeros(cat_dim,dtype=np.float32)
            unmatched+=1
            missing_by_cohort['input_date_not_in_v2']+=1
        else:
            num=transform['numeric'][j]
            mask=transform['numeric_mask'][j]
            cat=transform['categorical'][j]
            cat_mask=transform['categorical_mask'][j]
            matched+=1
        if sid not in first_seen:
            first_seen[sid]={'values':np.zeros(num_dim,dtype=np.float32),
                             'mask':np.zeros(num_dim,dtype=np.float32),
                             'date':scan}
        baseline=first_seen[sid]
        # First observed value per field is retained as baseline FROM THE MOMENT
        # IT IS ACTUALLY OBSERVED (no later-value backfill to earlier scans).
        newly=(baseline['mask']==0)&(mask>0)
        baseline['values'][newly]=num[newly]
        baseline['mask'][newly]=mask[newly]
        months=max((pd.Timestamp(scan)-pd.Timestamp(baseline['date'])).days/30.4375,0)
        # Clinical age is a *past-only* assessment lag from this DTI scan.
        # Use only tables represented by the selected feature set, and only
        # where an actual selected numeric field from that table is observed.
        # APOE/demographic constants and diagnosis history are not counted as
        # a current clinical assessment; otherwise old/static values would
        # falsely appear freshly observed.
        ages=[]
        if j is not None:
            table_prefixes={x.split('__',1)[0] for x in pp.numeric_fields
                            if x.startswith(('cognition__','function__','behavior__','other__'))}
            # Selected data schema is group__TABLE__field.
            table_names={x.split('__')[1] for x in pp.numeric_fields
                         if x.count('__')>=2 and x.split('__')[0] in
                         ('cognition','function','behavior','other')}
            for table in table_names:
                offset_col=f'{table}__offset_days'
                if offset_col not in visits:continue
                offset=pd.to_numeric(visits.iloc[j][offset_col],errors='coerce')
                if pd.notna(offset) and -126<=float(offset)<=0:
                    field_indices=[k for k,f in enumerate(pp.numeric_fields)
                                   if f.startswith(('cognition__'+table+'__',
                                                    'function__'+table+'__',
                                                    'behavior__'+table+'__',
                                                    'other__'+table+'__'))]
                    if any(mask[k]>0 for k in field_indices):ages.append(-float(offset))
        graph.meta_age_days=torch.tensor([min(ages) if ages else -1.0],dtype=torch.float32)
        graph.meta_num=torch.tensor(num.copy(),dtype=torch.float32)
        graph.meta_num_mask=torch.tensor(mask.copy(),dtype=torch.float32)
        graph.meta_cat=torch.tensor(cat.copy(),dtype=torch.long)
        graph.meta_present=torch.tensor([float(bool(mask.any() or cat_mask.any()))],dtype=torch.float32)
        graph.meta_baseline_num=torch.tensor(baseline['values'].copy(),dtype=torch.float32)
        graph.meta_baseline_mask=torch.tensor(baseline['mask'].copy(),dtype=torch.float32)
        graph.meta_months_since_baseline=torch.tensor([months],dtype=torch.float32)
        graph.meta_match_type='expanded_v2_fold'
    if matched==0:raise ValueError('No v2 clinical visit dates matched DTI scans; refusing training')
    fold_path=Path(fold_dir)
    fold_path.mkdir(parents=True,exist_ok=True)
    scaler_path=fold_path/'expanded_clinical_fold_scaler.json'
    pp.save(scaler_path)
    report={'alignment':alignment,'feature_set':feature_set,'numeric_features':num_dim,
            'categorical_features':cat_dim,'matched_graph_visits':matched,
            'unmatched_graph_visits':unmatched,'n_train_subjects':len(train_s),
            'n_val_subjects':len(val_s),'n_test_subjects':len(test_s),
            'scaler_path':str(scaler_path),'unused_future_diagnosis_as_input':True}
    (fold_path/'expanded_clinical_integration_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[expanded-clinical]',report)
    return pp,report
