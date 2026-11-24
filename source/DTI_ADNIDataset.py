#!/usr/bin/env python3
"""
DTI longitudinal ADNI dataset with visit-level DX and optional ADNIMERGE metadata.

This adapts the uploaded STGNN/fMRI project to your processed DTI ROI folders.
Each processed visit folder should look like:

  {dti_root}/{PTID}_{YYYY-MM-DD}/roi_stats/aal/roi_fa_stats.csv
  {dti_root}/{PTID}_{YYYY-MM-DD}/roi_stats/aal/roi_md_stats.csv
  {dti_root}/{PTID}_{YYYY-MM-DD}/roi_stats/aal/roi_rd_stats.csv
  {dti_root}/{PTID}_{YYYY-MM-DD}/roi_stats/aal/roi_ad_stats.csv

Each graph has:
  x: [116, 4] = mean FA/MD/RD/AD per AAL ROI
  edge_index: AAL spatial adjacency
  dx_label: visit-level diagnosis, CN=0, MCI=1, AD=2
  converter_label: subject-level converter/nonconverter from selected_300 CSV
  metadata: matched by PTID/EXAMDATE from ADNIMERGE or manifest
"""
from __future__ import annotations

import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

FOLDER_RE = re.compile(r"^(?P<subj>\d{3}_S_\d{4})_(?P<date>\d{4}-\d{2}-\d{2})$")
SCALARS = ["fa", "md", "rd", "ad"]

DX_MAP_3 = {
    "CN": 0, "NL": 0, "SMC": 0,
    "EMCI": 1, "LMCI": 1, "MCI": 1,
    "AD": 2, "Dementia": 2,
}
DX_CANONICAL = {
    "NL": "CN", "SMC": "CN",
    "CN": "CN",
    "EMCI": "MCI", "LMCI": "MCI", "MCI": "MCI",
    "AD": "AD", "Dementia": "AD",
}

META_CATEGORICAL_DEFAULT = [
    "PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4", "FLDSTRENG",
]

META_NUMERICAL_SAFE = [
    "AGE", "PTEDUCAT",
    "FDG", "PIB", "AV45",
    "ABETA", "TAU", "PTAU",
    "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV",
]

META_NUMERICAL_FULL = META_NUMERICAL_SAFE + [
    "CDRSB", "ADAS11", "ADAS13", "ADASQ4", "MMSE",
    "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
    "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ", "MOCA",
    "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
    "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal",
    "mPACCdigit", "mPACCtrailsB",
]

# Fallback aliases for manifest files with lowercase/simple covariates.
META_ALIASES = {
    "AGE": ["AGE", "age", "age_min_manifest"],
    "PTGENDER": ["PTGENDER", "sex", "gender"],
    "APOE4": ["APOE4", "apoe4"],
}


def _to_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def _normalize_date(x) -> Optional[str]:
    if pd.isna(x):
        return None
    dt = pd.to_datetime(str(x), errors="coerce")
    if pd.isna(dt):
        return None
    return str(dt.date())


def _months_between(date0: str, date1: str) -> float:
    d0 = pd.to_datetime(date0)
    d1 = pd.to_datetime(date1)
    return float((d1 - d0).days / 30.4375)


def _clean_dx(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return DX_CANONICAL.get(s, DX_CANONICAL.get(s.upper(), s))


def _dx_to_label(x) -> Optional[int]:
    s = _clean_dx(x)
    if s is None:
        return None
    return DX_MAP_3.get(s)


def _parse_numeric(x) -> float:
    """Parse ADNIMERGE biomarker strings such as '>1700' or '<8'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-4"}:
        return np.nan
    s = s.replace(",", "")
    if s.startswith(">") or s.startswith("<"):
        s = s[1:].strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_edge_graph(
    aal_adjacency: str | Path,
    add_self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load graph support and preserve numerical adjacency values as edge weights.

    Binary matrices produce unit weights. Weighted tractography matrices preserve
    their positive values. The matrix is symmetrized so each undirected tract has
    matching directed entries.
    """
    arr = np.load(str(Path(aal_adjacency).expanduser())).astype(np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = 0.5 * (arr + arr.T)
    if add_self_loops:
        diagonal = np.diag(arr).copy()
        diagonal[diagonal <= 0] = 1.0
        np.fill_diagonal(arr, diagonal)
    else:
        np.fill_diagonal(arr, 0.0)
    rows, cols = np.nonzero(arr > 0)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(arr[rows, cols], dtype=torch.float32)
    return edge_index, edge_weight


def load_edge_index(aal_adjacency: str | Path, add_self_loops: bool = False) -> torch.Tensor:
    """Backward-compatible support-only loader."""
    edge_index, _edge_weight = load_edge_graph(aal_adjacency, add_self_loops)
    return edge_index


def load_visit_x(visit_dir: Path, roi_subdir: str = "aal", n_rois: int = 116, clip_features: bool = True) -> torch.Tensor:
    roi_dir = visit_dir / "roi_stats" / roi_subdir
    if not roi_dir.exists():
        if roi_subdir != "aal":
            raise FileNotFoundError(f"Requested atlas ROI directory missing: {roi_dir}")
        roi_dir = visit_dir / "roi_stats"

    feats = []
    roi_ids_ref = None
    for scalar in SCALARS:
        p = roi_dir / f"roi_{scalar}_stats.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        df = pd.read_csv(p)
        if len(df) != n_rois:
            raise ValueError(f"{p} has {len(df)} rows, expected {n_rois}")
        if "roi_id" not in df.columns:
            raise ValueError(f"{p} missing roi_id column")
        mean_col = f"mean_{scalar}"
        if mean_col not in df.columns:
            raise ValueError(f"{p} missing {mean_col} column")
        df = df.sort_values("roi_id")
        roi_ids = df["roi_id"].astype(int).to_numpy()
        if roi_ids_ref is None:
            roi_ids_ref = roi_ids
        elif not np.array_equal(roi_ids_ref, roi_ids):
            raise ValueError(f"ROI id mismatch in {p}")
        feats.append(pd.to_numeric(df[mean_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))

    x = np.stack(feats, axis=1)  # [N, 4]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clip_features:
        x[:, 0] = np.clip(x[:, 0], 0.0, 1.0)      # FA
        x[:, 1] = np.clip(x[:, 1], 0.0, 0.004)    # MD
        x[:, 2] = np.clip(x[:, 2], 0.0, 0.004)    # RD
        x[:, 3] = np.clip(x[:, 3], 0.0, 0.004)    # AD
    return torch.tensor(x, dtype=torch.float32)


class MetadataTable:
    """Visit-level metadata matcher for ADNIMERGE/manifest rows."""
    def __init__(self, csv_path: Optional[str | Path], mode: str = "none", match_window_days: int = 126, scaler_checkpoint: Optional[str | Path] = None):
        self.csv_path = Path(csv_path).expanduser() if csv_path else None
        self.mode = mode
        self.match_window_days = int(match_window_days)
        self.scaler_checkpoint = Path(scaler_checkpoint).expanduser() if scaler_checkpoint else None
        self.df: Optional[pd.DataFrame] = None
        self.ptid_col: Optional[str] = None
        self.date_col: Optional[str] = None
        self.num_fields: List[str] = []
        self.baseline_columns: Dict[str, Optional[str]] = {}
        self.cat_fields: List[str] = []
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.num_mean: Optional[np.ndarray] = None
        self.num_std: Optional[np.ndarray] = None
        self._by_subject: Dict[str, pd.DataFrame] = {}
        self._empty_num = np.zeros((0,), dtype=np.float32)
        self._empty_mask = np.zeros((0,), dtype=np.float32)
        self._empty_cat = np.zeros((0,), dtype=np.int64)
        self.match_counts = Counter()

        if self.mode == "none" or self.csv_path is None:
            return
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")
        self._load()

    def _load(self) -> None:
        assert self.csv_path is not None
        df = pd.read_csv(self.csv_path, low_memory=False)
        ptid_col = _find_first_col(df, ["PTID", "subject_id", "Subject", "RID"])
        date_col = _find_first_col(df, ["EXAMDATE", "visit_date", "scan_date", "EXAM_DATE"])
        if ptid_col is None or date_col is None:
            raise ValueError(f"Metadata CSV needs PTID/subject_id and EXAMDATE/visit_date. Columns: {list(df.columns)[:30]}")
        df["_PTID"] = df[ptid_col].astype(str).str.strip()
        df["_DATE"] = df[date_col].map(_normalize_date)
        df = df[df["_PTID"].notna() & df["_DATE"].notna()].copy()
        df["_DATE_DT"] = pd.to_datetime(df["_DATE"])

        # Field selection: keep only fields present in this CSV, with simple aliases.
        wanted_num = META_NUMERICAL_SAFE if self.mode == "safe" else META_NUMERICAL_FULL
        num_fields = []
        for f in wanted_num:
            if f in df.columns:
                num_fields.append(f)
            elif f in META_ALIASES:
                c = _find_first_col(df, META_ALIASES[f])
                if c is not None and c not in num_fields:
                    num_fields.append(c)

        cat_fields = []
        for f in META_CATEGORICAL_DEFAULT:
            if f in df.columns:
                cat_fields.append(f)
            elif f in META_ALIASES:
                c = _find_first_col(df, META_ALIASES[f])
                if c is not None and c not in cat_fields:
                    cat_fields.append(c)

        # A pretrained imputer checkpoint may define the exact numerical field
        # order and scaler. This keeps pretraining and graph experiments on the
        # same numerical scale.
        scaler_blob = None
        if self.scaler_checkpoint is not None:
            if not self.scaler_checkpoint.exists():
                raise FileNotFoundError(f"Metadata scaler checkpoint not found: {self.scaler_checkpoint}")
            scaler_blob = torch.load(self.scaler_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_fields = list(scaler_blob.get("fields", []))
            if checkpoint_fields:
                missing_checkpoint_fields = [f for f in checkpoint_fields if f not in df.columns]
                if missing_checkpoint_fields:
                    raise ValueError(
                        "Metadata CSV is missing fields required by the imputer checkpoint: "
                        + ", ".join(missing_checkpoint_fields)
                    )
                num_fields = checkpoint_fields

        for f in num_fields:
            df[f"_NUM_{f}"] = df[f].map(_parse_numeric)
        num_arr = df[[f"_NUM_{f}" for f in num_fields]].to_numpy(dtype=np.float32) if num_fields else np.zeros((len(df), 0), dtype=np.float32)
        # Treat +/-inf metadata values as missing before scaling.
        if num_arr.size:
            num_arr[~np.isfinite(num_arr)] = np.nan
        if scaler_blob is not None and len(scaler_blob.get("mean", [])) == len(num_fields):
            mean = np.asarray(scaler_blob["mean"], dtype=np.float32)
            std = np.asarray(scaler_blob["std"], dtype=np.float32)
        else:
            mean = np.nanmean(num_arr, axis=0) if num_arr.shape[1] else np.zeros((0,), dtype=np.float32)
            std = np.nanstd(num_arr, axis=0) if num_arr.shape[1] else np.ones((0,), dtype=np.float32)
        mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
        std = np.where(np.nan_to_num(std, nan=1.0) < 1e-6, 1.0, np.nan_to_num(std, nan=1.0)).astype(np.float32)

        # Locate case-insensitive baseline partners, e.g. LDELTOTAL_BL.
        lower_to_actual = {str(c).lower(): str(c) for c in df.columns}
        baseline_columns = {
            f: lower_to_actual.get(f"{f}_bl".lower()) for f in num_fields
        }

        cat_maps = {}
        for f in cat_fields:
            vals = sorted(v for v in df[f].dropna().astype(str).str.strip().unique().tolist() if v != "")
            cat_maps[f] = {v: i for i, v in enumerate(vals)}  # missing = len(map)

        self.df = df
        self.ptid_col = ptid_col
        self.date_col = date_col
        self.num_fields = num_fields
        self.baseline_columns = baseline_columns
        self.cat_fields = cat_fields
        self.cat_maps = cat_maps
        self.num_mean = mean
        self.num_std = std
        self._by_subject = {s: sub.sort_values("_DATE_DT").copy() for s, sub in df.groupby("_PTID")}

    @property
    def num_dim(self) -> int:
        return len(self.num_fields)

    @property
    def cat_cardinalities(self) -> List[int]:
        return [len(self.cat_maps[f]) + 1 for f in self.cat_fields]

    def get(self, subject_id: str, visit_date: str):
        if self.mode == "none" or self.df is None:
            return (
                self._empty_num, self._empty_mask, self._empty_cat,
                self._empty_num, self._empty_mask, np.zeros((1,), dtype=np.float32),
                False, "none",
            )
        sub = self._by_subject.get(subject_id)
        if sub is None or sub.empty:
            self.match_counts["missing_subject"] += 1
            return *self._missing(), False, "missing_subject"
        target_dt = pd.to_datetime(visit_date)
        exact = sub[sub["_DATE"] == visit_date]
        if not exact.empty:
            row = exact.iloc[0]
            self.match_counts["exact"] += 1
            return *self._row_to_arrays(row, target_dt), True, "exact"
        # Forecasting-safe fallback: use only the most recent metadata row on or
        # before the DTI date. Never match a future clinical visit.
        previous = sub[sub["_DATE_DT"] <= target_dt]
        if previous.empty:
            self.match_counts["no_previous"] += 1
            return *self._baseline_only(sub.iloc[0], target_dt), False, "no_previous"
        row = previous.iloc[-1]
        days = int((target_dt - row["_DATE_DT"]).days)
        if days <= self.match_window_days:
            self.match_counts["previous"] += 1
            return *self._row_to_arrays(row, target_dt), True, f"previous_{days}d"
        self.match_counts["outside_window"] += 1
        return *self._baseline_only(row, target_dt), False, f"outside_window_{days}d"

    def _baseline_only(self, row: pd.Series, target_dt: pd.Timestamp):
        arrays = self._row_to_arrays(row, target_dt)
        _nums, _mask, _cats, baseline_nums, baseline_mask, months = arrays
        # If the claimed baseline occurs after prediction time, discard it.
        baseline_date_col = next((c for c in row.index if str(c).lower() == "examdate_bl"), None)
        baseline_dt = pd.to_datetime(row.get(baseline_date_col), errors="coerce") if baseline_date_col else pd.NaT
        if pd.isna(baseline_dt) or baseline_dt > target_dt:
            baseline_nums = np.zeros((self.num_dim,), dtype=np.float32)
            baseline_mask = np.zeros((self.num_dim,), dtype=np.float32)
            months = np.zeros((1,), dtype=np.float32)
        return (
            np.zeros((self.num_dim,), dtype=np.float32),
            np.zeros((self.num_dim,), dtype=np.float32),
            np.array([self.cat_cardinalities[i] - 1 for i in range(len(self.cat_fields))], dtype=np.int64),
            baseline_nums, baseline_mask, months,
        )

    def _missing(self):
        return (
            np.zeros((self.num_dim,), dtype=np.float32),
            np.zeros((self.num_dim,), dtype=np.float32),
            np.array([self.cat_cardinalities[i] - 1 for i in range(len(self.cat_fields))], dtype=np.int64),
            np.zeros((self.num_dim,), dtype=np.float32),
            np.zeros((self.num_dim,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
        )

    def _row_to_arrays(self, row: pd.Series, target_dt: pd.Timestamp):
        nums, mask = [], []
        baseline_nums, baseline_mask = [], []
        assert self.num_mean is not None and self.num_std is not None
        for i, f in enumerate(self.num_fields):
            v = _parse_numeric(row.get(f, np.nan))
            if not np.isfinite(v):
                nums.append(0.0); mask.append(0.0)
            else:
                z = (float(v) - float(self.num_mean[i])) / float(self.num_std[i])
                if not np.isfinite(z):
                    nums.append(0.0)
                    mask.append(0.0)
                else:
                    nums.append(float(z))
                    mask.append(1.0)

            baseline_col = self.baseline_columns.get(f)
            bv = _parse_numeric(row.get(baseline_col, np.nan)) if baseline_col else np.nan
            if np.isnan(bv):
                baseline_nums.append(0.0); baseline_mask.append(0.0)
            else:
                baseline_nums.append((float(bv) - float(self.num_mean[i])) / float(self.num_std[i]))
                baseline_mask.append(1.0)

        cats = []
        for f in self.cat_fields:
            value = "" if pd.isna(row.get(f, np.nan)) else str(row.get(f)).strip()
            cats.append(self.cat_maps[f].get(value, len(self.cat_maps[f])))

        baseline_date_col = next((c for c in row.index if str(c).lower() == "examdate_bl"), None)
        baseline_dt = pd.to_datetime(row.get(baseline_date_col), errors="coerce") if baseline_date_col else pd.NaT
        months_since_baseline = 0.0
        if pd.notna(baseline_dt):
            months_since_baseline = max(float((target_dt - baseline_dt).days / 30.4375), 0.0)

        return (
            np.asarray(nums, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(cats, dtype=np.int64),
            np.asarray(baseline_nums, dtype=np.float32),
            np.asarray(baseline_mask, dtype=np.float32),
            np.asarray([months_since_baseline], dtype=np.float32),
        )


class VisitLabelTable:
    """Visit-level diagnosis matcher from manifest or ADNIMERGE."""
    def __init__(self, manifest_csv: str | Path, match_window_days: int = 126):
        self.manifest_csv = Path(manifest_csv).expanduser()
        self.match_window_days = int(match_window_days)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv, low_memory=False)
        ptid_col = _find_first_col(df, ["subject_id", "PTID", "Subject"])
        date_col = _find_first_col(df, ["visit_date", "EXAMDATE", "scan_date"])
        dx_col = _find_first_col(df, ["diagnosis", "DX", "DX_bl", "dx"])
        label_col = _find_first_col(df, ["label"])
        if ptid_col is None or date_col is None:
            raise ValueError("Manifest needs subject_id/PTID and visit_date/EXAMDATE columns")
        if dx_col is None and label_col is None:
            raise ValueError("Manifest needs diagnosis/DX or label column")
        df["_PTID"] = df[ptid_col].astype(str).str.strip()
        df["_DATE"] = df[date_col].map(_normalize_date)
        if dx_col is not None:
            df["_DX_STR"] = df[dx_col].map(_clean_dx)
            df["_DX_LABEL"] = df[dx_col].map(_dx_to_label)
        else:
            df["_DX_STR"] = None
            df["_DX_LABEL"] = pd.to_numeric(df[label_col], errors="coerce")
        df = df[df["_PTID"].notna() & df["_DATE"].notna() & df["_DX_LABEL"].notna()].copy()
        df["_DX_LABEL"] = df["_DX_LABEL"].astype(int)
        # normalize label if needed: assume already 0/1/2 if label col.
        df["_DATE_DT"] = pd.to_datetime(df["_DATE"])
        self.df = df
        self.ptid_col = ptid_col
        self.date_col = date_col
        self.dx_col = dx_col
        self.label_col = label_col
        self._by_subject = {s: sub.sort_values("_DATE_DT").copy() for s, sub in df.groupby("_PTID")}
        self.match_counts = Counter()

    def get(self, subject_id: str, visit_date: str) -> Optional[Dict[str, Any]]:
        sub = self._by_subject.get(subject_id)
        if sub is None or sub.empty:
            self.match_counts["missing_subject"] += 1
            return None
        exact = sub[sub["_DATE"] == visit_date]
        if not exact.empty:
            row = exact.iloc[0]
            self.match_counts["exact"] += 1
            return self._row_to_dict(row, "exact", 0)
        target_dt = pd.to_datetime(visit_date)
        diffs = (sub["_DATE_DT"] - target_dt).abs().dt.days
        j = int(diffs.argmin())
        days = int(diffs.iloc[j])
        if days <= self.match_window_days:
            row = sub.iloc[j]
            self.match_counts["nearest"] += 1
            return self._row_to_dict(row, "nearest", days)
        self.match_counts["outside_window"] += 1
        return None

    def _row_to_dict(self, row: pd.Series, kind: str, days: int) -> Dict[str, Any]:
        label = int(row["_DX_LABEL"])
        dx_str = row.get("_DX_STR")
        if not dx_str or pd.isna(dx_str):
            dx_str = {0: "CN", 1: "MCI", 2: "AD"}.get(label, str(label))
        return {
            "dx_label": label,
            "dx_str": str(dx_str),
            "matched_date": str(row["_DATE"]),
            "match_type": kind,
            "date_diff_days": int(days),
        }


class DTI_ADNIDataset(Dataset):
    def __init__(
        self,
        dti_root: str | Path,
        selected_subjects_csv: str | Path,
        aal_adjacency: str | Path,
        manifest_csv: str | Path,
        meta_csv: Optional[str | Path] = None,
        meta_mode: str = "none",
        roi_subdir: str = "aal",
        min_visits: int = 2,
        max_visits: int = 10,
        n_rois: int = 116,
        standardize: bool = True,
        match_window_days: int = 126,
        clip_features: bool = True,
        ollama_cache: Optional[str | Path] = None,
        meta_scaler_checkpoint: Optional[str | Path] = None,
    ):
        super().__init__()
        self.dti_root = Path(dti_root).expanduser()
        self.selected_subjects_csv = Path(selected_subjects_csv).expanduser()
        self.aal_adjacency = Path(aal_adjacency).expanduser()
        self.manifest_csv = Path(manifest_csv).expanduser()
        self.meta_csv = Path(meta_csv).expanduser() if meta_csv else None
        self.meta_mode = meta_mode
        self.roi_subdir = roi_subdir
        self.min_visits = int(min_visits)
        self.max_visits = int(max_visits)
        self.n_rois = int(n_rois)
        self.standardize = standardize
        self.match_window_days = int(match_window_days)
        self.clip_features = clip_features
        self.ollama_cache_path = Path(ollama_cache).expanduser() if ollama_cache else None
        self.meta_scaler_checkpoint = Path(meta_scaler_checkpoint).expanduser() if meta_scaler_checkpoint else None
        self.ollama_embeddings = {}
        self.ollama_dim = 0
        self.ollama_match_counts = Counter()
        self._load_ollama_cache()

        for p, name in [(self.dti_root, "DTI root"), (self.selected_subjects_csv, "selected CSV"), (self.aal_adjacency, "AAL adjacency"), (self.manifest_csv, "manifest CSV")]:
            if not p.exists():
                raise FileNotFoundError(f"{name} does not exist: {p}")

        self.edge_index, self.edge_weight = load_edge_graph(self.aal_adjacency)
        self.subject_table = self._load_subject_table()
        self.label_table = VisitLabelTable(self.manifest_csv, match_window_days=self.match_window_days)
        meta_source = self.meta_csv if self.meta_csv and self.meta_csv.exists() else self.manifest_csv
        self.meta_table = MetadataTable(
            meta_source, mode=self.meta_mode, match_window_days=self.match_window_days,
            scaler_checkpoint=self.meta_scaler_checkpoint,
        )
        self.data_list: List[Data] = []
        self.subject_graph_dict: Dict[str, List[Data]] = {}
        self.skipped: Dict[str, int] = defaultdict(int)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self._build()
        if self.standardize and self.data_list:
            self._standardize_features()

    def _load_ollama_cache(self) -> None:
        """Load cached Ollama embeddings keyed by (subject_id, visit_date)."""
        if self.ollama_cache_path is None:
            return
        if not self.ollama_cache_path.exists():
            raise FileNotFoundError(f"Ollama cache not found: {self.ollama_cache_path}")

        blob = torch.load(self.ollama_cache_path, map_location="cpu", weights_only=False)
        emb_dict = blob.get("embeddings", blob)

        cleaned = {}
        dim = int(blob.get("dim", 0)) if isinstance(blob, dict) else 0

        for k, v in emb_dict.items():
            # Accept tuple keys, list keys, or string keys like "PTID|YYYY-MM-DD".
            if isinstance(k, tuple) and len(k) == 2:
                subj, date = str(k[0]), str(k[1])
            elif isinstance(k, list) and len(k) == 2:
                subj, date = str(k[0]), str(k[1])
            else:
                ks = str(k)
                if "|" in ks:
                    subj, date = ks.split("|", 1)
                elif "::" in ks:
                    subj, date = ks.split("::", 1)
                else:
                    continue

            t = torch.as_tensor(v, dtype=torch.float32).view(-1)
            if dim <= 0:
                dim = int(t.numel())
            cleaned[(subj, date)] = t

        self.ollama_embeddings = cleaned
        self.ollama_dim = int(dim)

    def _get_ollama_embedding(self, subject_id: str, visit_date: str) -> Tuple[torch.Tensor, bool]:
        if self.ollama_dim <= 0:
            return torch.zeros((0,), dtype=torch.float32), False
        v = self.ollama_embeddings.get((subject_id, visit_date))
        if v is None:
            self.ollama_match_counts["missing"] += 1
            return torch.zeros((self.ollama_dim,), dtype=torch.float32), False
        self.ollama_match_counts["exact"] += 1
        return v.float(), True

    @property
    def meta_num_dim(self) -> int:
        return self.meta_table.num_dim

    @property
    def meta_cat_cardinalities(self) -> List[int]:
        return self.meta_table.cat_cardinalities

    def _load_subject_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.selected_subjects_csv, low_memory=False)
        if "subject_id" not in df.columns:
            raise ValueError("selected_subjects_csv must have subject_id column")
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
        if "is_converter" in df.columns:
            df["converter_label"] = df["is_converter"].map(_to_bool).astype(int)
        elif "converter_group" in df.columns:
            df["converter_label"] = (df["converter_group"].astype(str).str.lower() == "converter").astype(int)
        else:
            raise ValueError("selected CSV needs is_converter or converter_group")
        return df

    def _build(self) -> None:
        converter_by_subject = dict(zip(self.subject_table["subject_id"], self.subject_table["converter_label"]))
        selected_ids = set(converter_by_subject)

        visit_records = []
        for vdir in sorted(self.dti_root.iterdir()):
            if not vdir.is_dir():
                continue
            m = FOLDER_RE.match(vdir.name)
            if not m:
                self.skipped["bad_folder_name"] += 1
                continue
            subj = m.group("subj")
            date = m.group("date")
            if subj not in selected_ids:
                self.skipped["not_selected_subject"] += 1
                continue
            try:
                _ = load_visit_x(vdir, self.roi_subdir, self.n_rois, self.clip_features)
            except Exception:
                self.skipped["missing_or_bad_roi"] += 1
                continue
            visit_label = self.label_table.get(subj, date)
            if visit_label is None or visit_label["dx_label"] not in {0, 1, 2}:
                self.skipped["no_visit_dx"] += 1
                continue
            visit_records.append((subj, date, vdir, visit_label))

        by_subject = defaultdict(list)
        for rec in visit_records:
            by_subject[rec[0]].append(rec)

        for subj, visits in sorted(by_subject.items()):
            visits = sorted(visits, key=lambda t: t[1])
            if len(visits) < self.min_visits:
                self.skipped["too_few_visits"] += 1
                continue
            if self.max_visits and len(visits) > self.max_visits:
                visits = visits[-self.max_visits:]

            first_date = visits[0][1]
            graphs = []
            for j, (subj, date, vdir, visit_label) in enumerate(visits, start=1):
                x = load_visit_x(vdir, self.roi_subdir, self.n_rois, self.clip_features)
                dx_label = int(visit_label["dx_label"])
                converter_label = int(converter_by_subject[subj])
                visit_months = torch.tensor([_months_between(first_date, date)], dtype=torch.float32)
                if j < len(visits):
                    next_date = visits[j][1]
                    months_to_next = torch.tensor([_months_between(date, next_date)], dtype=torch.float32)
                else:
                    months_to_next = torch.tensor([float("nan")], dtype=torch.float32)

                meta_result = self.meta_table.get(subj, date)
                if len(meta_result) != 8:
                    raise ValueError(f"MetadataTable.get returned {len(meta_result)} values; expected 8. Value={meta_result}")
                (
                    meta_num, meta_mask, meta_cat, meta_baseline_num,
                    meta_baseline_mask, meta_months_since_baseline,
                    meta_present, meta_match_type,
                ) = meta_result

                data = Data(x=x, edge_index=self.edge_index.clone(), edge_weight=self.edge_weight.clone(), y=torch.tensor([dx_label], dtype=torch.long))
                data.dx_label = torch.tensor([dx_label], dtype=torch.long)
                data.converter_label = torch.tensor([converter_label], dtype=torch.long)
                data.meta_num = torch.tensor(meta_num, dtype=torch.float32)
                data.meta_num_mask = torch.tensor(meta_mask, dtype=torch.float32)
                data.meta_cat = torch.tensor(meta_cat, dtype=torch.long)
                data.meta_baseline_num = torch.tensor(meta_baseline_num, dtype=torch.float32)
                data.meta_baseline_mask = torch.tensor(meta_baseline_mask, dtype=torch.float32)
                data.meta_months_since_baseline = torch.tensor(meta_months_since_baseline, dtype=torch.float32)
                data.meta_present = torch.tensor([1.0 if meta_present else 0.0], dtype=torch.float32)
                ollama_emb, ollama_present = self._get_ollama_embedding(subj, date)
                data.ollama_emb = ollama_emb
                data.ollama_present = torch.tensor([1.0 if ollama_present else 0.0], dtype=torch.float32)
                data.subj_id = f"{subj}_run{j:02d}"
                data.subject_id = subj
                data.visit_date = date
                data.visit_order = torch.tensor([j], dtype=torch.long)
                data.visit_months = visit_months
                data.months_to_next = months_to_next
                data.visit_dir = str(vdir)
                data.dx_str = visit_label["dx_str"]
                data.dx_match_type = visit_label["match_type"]
                data.dx_date_diff_days = int(visit_label["date_diff_days"])
                data.meta_match_type = meta_match_type
                graphs.append(data)
                self.data_list.append(data)
            self.subject_graph_dict[subj] = graphs

    def _standardize_features(self) -> None:
        xs = torch.cat([d.x for d in self.data_list], dim=0)
        mean = xs.mean(dim=0, keepdim=True)
        std = xs.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        self.feature_mean = mean.squeeze(0)
        self.feature_std = std.squeeze(0)
        for d in self.data_list:
            d.x = (d.x - mean) / std

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    def subject_split_labels(self, task: str = "next_dx") -> Dict[str, int]:
        labels = {}
        for s, graphs in self.subject_graph_dict.items():
            if not graphs:
                continue
            if task == "converter":
                labels[s] = int(graphs[0].converter_label.item())
            else:
                labels[s] = int(graphs[-1].dx_label.item())
        return labels

    def summary(self) -> Dict:
        subject_converter = {s: int(g[0].converter_label.item()) for s, g in self.subject_graph_dict.items() if g}
        subject_last_dx = {s: int(g[-1].dx_label.item()) for s, g in self.subject_graph_dict.items() if g}
        visit_dx = [int(d.dx_label.item()) for d in self.data_list]
        visit_counts = {s: len(g) for s, g in self.subject_graph_dict.items()}
        return {
            "n_graphs_visits": len(self.data_list),
            "n_subjects": len(self.subject_graph_dict),
            "converter_counts_subject_level": {str(k): int(v) for k, v in pd.Series(subject_converter).value_counts().sort_index().to_dict().items()},
            "last_dx_counts_subject_level": {str(k): int(v) for k, v in pd.Series(subject_last_dx).value_counts().sort_index().to_dict().items()},
            "visit_dx_counts": {str(k): int(v) for k, v in pd.Series(visit_dx).value_counts().sort_index().to_dict().items()},
            "dx_label_map": {"CN": 0, "MCI": 1, "AD": 2},
            "visit_count_min": int(min(visit_counts.values())) if visit_counts else 0,
            "visit_count_median": float(np.median(list(visit_counts.values()))) if visit_counts else 0.0,
            "visit_count_mean": float(np.mean(list(visit_counts.values()))) if visit_counts else 0.0,
            "visit_count_max": int(max(visit_counts.values())) if visit_counts else 0,
            "skipped": dict(self.skipped),
            "label_match_counts": dict(self.label_table.match_counts),
            "metadata_mode": self.meta_mode,
            "metadata_source": str(self.meta_table.csv_path) if self.meta_table.csv_path else None,
            "metadata_num_fields": self.meta_table.num_fields,
            "metadata_cat_fields": self.meta_table.cat_fields,
            "metadata_match_counts": dict(self.meta_table.match_counts),
            "ollama_cache": str(self.ollama_cache_path) if self.ollama_cache_path else None,
            "ollama_dim": int(self.ollama_dim),
            "ollama_match_counts": dict(self.ollama_match_counts),
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "adjacency_directed_edges": int(self.edge_index.size(1)),
            "adjacency_weight_min": float(self.edge_weight.min().item()) if self.edge_weight.numel() else 0.0,
            "adjacency_weight_mean": float(self.edge_weight.mean().item()) if self.edge_weight.numel() else 0.0,
            "adjacency_weight_max": float(self.edge_weight.max().item()) if self.edge_weight.numel() else 0.0,
        }

    def to_subject_visit_frame(self) -> pd.DataFrame:
        rows = []
        for s, graphs in self.subject_graph_dict.items():
            for d in graphs:
                rows.append({
                    "subject_id": s,
                    "subj_id": d.subj_id,
                    "visit_date": d.visit_date,
                    "visit_order": int(d.visit_order.item()),
                    "dx_label": int(d.dx_label.item()),
                    "dx_str": d.dx_str,
                    "converter_label": int(d.converter_label.item()),
                    "dx_match_type": d.dx_match_type,
                    "dx_date_diff_days": d.dx_date_diff_days,
                    "meta_match_type": d.meta_match_type,
                    "ollama_present": float(d.ollama_present.item()) if hasattr(d, "ollama_present") else 0.0,
                    "visit_dir": d.visit_dir,
                })
        return pd.DataFrame(rows)
