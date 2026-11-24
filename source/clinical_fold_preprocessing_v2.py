#!/usr/bin/env python3
"""Fold-safe preprocessing adapter for clinical features produced by build_expanded_clinical_dataset.py.

1. Fit exclusively on DISTINCT INPUT DTI VISITS from the TRAIN SUBJECT set.
2. Return (standardized numeric values, numeric observation masks,
   integer categorical codes, categorical masks) for each 2D batch/sequence.
3. Do not include target outcomes or train/test-fold-dependent information
   in the feature schema.

Imported into the training loop AFTER subject-level fold assignment. The
builder outputs RAW values; do not fit global scalers on all patients.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_CATEGORICAL={'context__DXSUM__diagnosis','context__PTDEMOG__PTGENDER'}


def _list_features(schema, feature_set):
    names=schema['ordered_features']
    if feature_set not in schema['feature_sets']:
        raise ValueError(f'Unknown feature set {feature_set}; expected {list(schema["feature_sets"])}')
    return [names[i] for i in schema['feature_sets'][feature_set]]


@dataclass
class ClinicalFoldPreprocessor:
    selected_features: list[str]
    numeric_fields: list[str]
    categorical_fields: list[str]
    num_mean: np.ndarray
    num_std: np.ndarray
    categorical_vocab: dict[str,dict[str,int]]
    feature_set: str
    training_subject_count: int
    training_visit_count: int

    @classmethod
    def fit(cls, visit_csv: str|Path, feature_schema: str|Path,
            train_subjects: set[str]|list[str], feature_set='C2_primary'):
        schema=json.loads(Path(feature_schema).read_text())
        selected=_list_features(schema,feature_set)
        ids=set(map(str,train_subjects))
        if not ids:raise ValueError('train_subjects must contain at least one ID')
        df=pd.read_csv(visit_csv,low_memory=False)
        if 'subject_id' not in df:raise ValueError('visit CSV must have subject_id')
        train=df.loc[df.subject_id.astype(str).isin(ids)].copy()
        if train.empty:raise ValueError('No training visits match train_subjects')
        # Guard against unexpected duplicate visit IDs.
        if train.duplicated(['subject_id','scan_date']).any():
            raise ValueError('Duplicate train visits, expected one per input DTI scan')
        missing=set(selected)-set(df.columns)
        if missing:raise ValueError(f'Schema-feature mismatch: missing {sorted(missing)[:15]}')
        cat=[f for f in selected if f in DEFAULT_CATEGORICAL]
        nums=[f for f in selected if f not in DEFAULT_CATEGORICAL]
        X=train[nums].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        X[~np.isfinite(X)]=np.nan
        valid=np.isfinite(X)
        means=np.array([float(np.mean(X[valid[:,i],i])) if np.any(valid[:,i]) else 0.0
                        for i in range(X.shape[1])],dtype=np.float32)
        stds=np.array([float(np.std(X[valid[:,i],i])) if np.any(valid[:,i]) else 1.0
                       for i in range(X.shape[1])],dtype=np.float32)
        stds=np.where(stds>1e-6,stds,1.0)
        vocab={}
        for c in cat:
            s=pd.to_numeric(train[c],errors='coerce')
            unique=sorted(set(float(v) for v in s[np.isfinite(s)]))
            vocab[c]={str(v):j+1 for j,v in enumerate(unique)}
        return cls(selected,nums,cat,means,stds,vocab,feature_set,len(ids),len(train))

    def transform_dataframe(self, frame:pd.DataFrame):
        """Transform visits or last-input rows: never refit on held-out data."""
        missing=set(self.selected_features)-set(frame.columns)
        if missing:raise ValueError(f'Missing expected columns: {sorted(missing)}')
        num=frame[self.numeric_fields].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
        obs=np.isfinite(num)
        normalized=np.where(obs,(num-self.num_mean)/self.num_std,0.0).astype('float32')
        cat=np.zeros((len(frame),len(self.categorical_fields)),dtype=np.int64)
        c_mask=np.zeros_like(cat,dtype=np.float32)
        for i,c in enumerate(self.categorical_fields):
            values=pd.to_numeric(frame[c],errors='coerce')
            for j,v in enumerate(values):
                if pd.notna(v) and np.isfinite(v):
                    code=self.categorical_vocab[c].get(str(float(v)),0)
                    cat[j,i]=code
                    if code:c_mask[j,i]=1
        return {'numeric':normalized,'numeric_mask':obs.astype('float32'),
                'categorical':cat,'categorical_mask':c_mask}

    def transform_sequence(self, values, masks, all_feature_names):
        """Transform dense JSONL list values/masks, with variable sequence length."""
        v=np.asarray(values,dtype=np.float32)
        m=np.asarray(masks,dtype=bool)
        if v.ndim!=2 or v.shape!=m.shape or v.shape[1]!=len(all_feature_names):
            raise ValueError('Sequence shape/mask/schema mismatch')
        df=pd.DataFrame(np.where(m,v,np.nan),columns=all_feature_names)
        return self.transform_dataframe(df)

    def save(self,path:str|Path):
        out={'selected_features':self.selected_features,
             'numeric_fields':self.numeric_fields,'categorical_fields':self.categorical_fields,
             'num_mean':self.num_mean.tolist(),'num_std':self.num_std.tolist(),
             'categorical_vocab':self.categorical_vocab,'feature_set':self.feature_set,
             'training_subject_count':self.training_subject_count,
             'training_visit_count':self.training_visit_count}
        p=Path(path)
        with p.open('w') as f:json.dump(out,f,indent=2)

    @classmethod
    def load(cls,path:str|Path):
        o=json.loads(Path(path).read_text())
        o['num_mean']=np.asarray(o['num_mean'],dtype=np.float32)
        o['num_std']=np.asarray(o['num_std'],dtype=np.float32)
        return cls(**o)

    @property
    def categorical_cardinalities(self):
        return [len(self.categorical_vocab[f])+1 for f in self.categorical_fields]


def assert_disjoint_folds(train_subjects,val_subjects,test_subjects):
    a,b,c=map(lambda x:set(map(str,x)),(train_subjects,val_subjects,test_subjects))
    if a&b or a&c or b&c:raise ValueError('Subject-level train/val/test overlap detected')

