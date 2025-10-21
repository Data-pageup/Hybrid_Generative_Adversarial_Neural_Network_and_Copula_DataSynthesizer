"""
synth_gen.py

Updated core module for tabular synthetic dataset generation.
Supports SDV >=1.26 (single_table API) when available and provides a lightweight fallback
when SDV is not installed. Exported functions used by Streamlit app:

- infer_column_types
- preprocess_for_sdv
- build_metadata_from_df
- fit_model / load_model
- generate_from_model
- evaluate_synthetic
- memorization_check

Dependencies: pandas, numpy, scipy, scikit-learn, joblib; sdv optional but recommended.
"""

import os
import uuid
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

MODEL_DIR = os.environ.get("SYNTH_MODEL_DIR", "/tmp/synth_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Try to import SDV (single_table API). If unavailable, provide a lightweight fallback
# ---------------------------------------------------------------------------
_HAVE_SDV = False
try:
    # SDV >=1.26 style imports
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata.single_table import SingleTableMetadata
    _HAVE_SDV = True
except Exception:
    _HAVE_SDV = False

    class SingleTableMetadata:
        """Dummy metadata placeholder for fallback: only to keep signatures compatible."""
        def __init__(self):
            pass
        def detect_from_dataframe(self, df):
            return None

    class BasicSynthModel:
        """Lightweight fallback synthesizer for prototyping (marginals + bootstrap noise).

        This does NOT model joint distributions correctly but is useful for prototyping
        when SDV isn't installed.
        """
        def __init__(self, metadata=None, noise_frac: float = 0.01):
            self._df = None
            self.noise_frac = noise_frac

        def fit(self, df: pd.DataFrame):
            self._df = df.copy()
            return self

        def sample(self, num_rows: int = 100):
            if self._df is None:
                raise RuntimeError("Model not fitted")
            n = int(num_rows)
            out = {}
            for c in self._df.columns:
                s = self._df[c].dropna()
                if pd.api.types.is_numeric_dtype(s):
                    if len(s) == 0:
                        out[c] = np.full(n, np.nan)
                    else:
                        samp = s.sample(n=n, replace=True).to_numpy(dtype=float)
                        std = np.nanstd(samp)
                        noise_scale = (std if std > 0 else 1.0) * self.noise_frac
                        samp = samp + np.random.normal(0, noise_scale, size=n)
                        out[c] = samp
                elif pd.api.types.is_datetime64_any_dtype(s):
                    try:
                        s_ts = pd.to_datetime(s)
                        min_ts = s_ts.min().value
                        max_ts = s_ts.max().value
                        rand = np.random.randint(min_ts, max_ts + 1, size=n)
                        out[c] = pd.to_datetime(rand)
                    except Exception:
                        out[c] = [pd.NaT] * n
                else:
                    if len(s) == 0:
                        out[c] = [None] * n
                    else:
                        counts = s.value_counts(normalize=True)
                        cats = counts.index.to_list()
                        probs = counts.values
                        out[c] = np.random.choice(cats, size=n, p=probs)
            return pd.DataFrame(out)

    # Map synthesizer names to fallback so rest of code can call them
    GaussianCopulaSynthesizer = BasicSynthModel
    CTGANSynthesizer = BasicSynthModel

# ---------------------------------------------------------------------------
# Schema inference & preprocessing
# ---------------------------------------------------------------------------

def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer column types as one of: 'numerical','categorical','datetime','bool','high_cardinality'.

    Heuristics:
    - datetime dtype -> 'datetime'
    - bool dtype or only 0/1 values -> 'bool'
    - numeric: low unique -> 'categorical' else 'numerical'
    - object: treat as categorical, but mark high_cardinality if many uniques
    """
    types: Dict[str, str] = {}
    n = len(df)
    for c in df.columns:
        series = df[c]
        try:
            if pd.api.types.is_bool_dtype(series) or series.dropna().isin([0, 1]).all():
                types[c] = 'bool'
            elif pd.api.types.is_datetime64_any_dtype(series):
                types[c] = 'datetime'
            elif pd.api.types.is_numeric_dtype(series):
                uniq = int(series.nunique(dropna=True))
                if uniq < max(20, int(0.02 * max(1, n))):
                    types[c] = 'categorical'
                else:
                    types[c] = 'numerical'
            else:
                uniq = int(series.nunique(dropna=True))
                if uniq > 500 and n > 10000:
                    types[c] = 'high_cardinality'
                else:
                    types[c] = 'categorical'
        except Exception:
            types[c] = 'categorical'
    return types


def preprocess_for_sdv(df: pd.DataFrame, coltypes: Dict[str, str], top_k: int = 500) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare dataframe for SDV-style models.

    - Cast object-like categorical columns to 'category'
    - For high-cardinality columns, keep top_k categories and map rest to a special token
    - Convert datetime-like to datetime dtype

    Returns processed df and a 'meta' dict to support postprocessing
    """
    df2 = df.copy()
    meta: Dict[str, Any] = {'coltypes': coltypes, 'categories': {}}

    for c, t in coltypes.items():
        if c not in df2.columns:
            continue
        if t == 'datetime':
            df2[c] = pd.to_datetime(df2[c], errors='coerce')
        elif t in ('categorical', 'bool', 'high_cardinality'):
            if t == 'high_cardinality':
                counts = df2[c].value_counts(dropna=True)
                top = set(counts.index[:top_k])
                other_label = f"__OTHER__{c}"
                df2[c] = df2[c].where(df2[c].isin(top), other_label)
            df2[c] = df2[c].astype('category')
            meta['categories'][c] = list(df2[c].cat.categories)
        else:
            # numerical left as-is
            pass
    return df2, meta

# ---------------------------------------------------------------------------
# SDV metadata / model helpers
# ---------------------------------------------------------------------------

def build_metadata_from_df(df: pd.DataFrame) -> SingleTableMetadata:
    """Create SingleTableMetadata from dataframe (SDV) or return dummy metadata for fallback."""
    if _HAVE_SDV:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        return metadata
    else:
        m = SingleTableMetadata()
        try:
            m.detect_from_dataframe(df)
        except Exception:
            pass
        return m


def _model_path(job_id: str) -> str:
    return os.path.join(MODEL_DIR, f"{job_id}.pkl")


def fit_model(df: pd.DataFrame, model_type: str = 'gaussiancopula', job_id: str = None, metadata: SingleTableMetadata = None, **kwargs) -> str:
    """Fit an SDV single-table synthesizer (or fallback) and persist both synthesizer and metadata.

    Returns job_id.
    """
    if job_id is None:
        job_id = str(uuid.uuid4())

    if metadata is None:
        metadata = build_metadata_from_df(df)

    mt = model_type.lower()
    if mt in ('gaussiancopula', 'gaussiancopula_synth', 'gaussiancopula_synthesizer'):
        SynthClass = GaussianCopulaSynthesizer
    elif mt in ('ctgan', 'ctgan_synth', 'ctgan_synthesizer'):
        SynthClass = CTGANSynthesizer
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # instantiate with metadata if SDV available, else fallback ignores metadata
    try:
        synth = SynthClass(metadata, **kwargs) if _HAVE_SDV else SynthClass(metadata, **{'noise_frac': kwargs.get('noise_frac', 0.01)})
    except TypeError:
        # Some fallback or older classes may accept no args
        synth = SynthClass()

    # Fit and persist
    synth.fit(df)
    joblib.dump({'synthesizer': synth, 'metadata': metadata}, _model_path(job_id))
    return job_id


def load_model(job_id: str):
    path = _model_path(job_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {job_id}")
    data = joblib.load(path)
    if isinstance(data, dict) and 'synthesizer' in data:
        return data['synthesizer'], data.get('metadata', None)
    return data, None


def generate_from_model(job_id: str, n_rows: int) -> pd.DataFrame:
    synth, metadata = load_model(job_id)
    # SDV uses sample(num_rows=...); fallback uses sample(num_rows)
    try:
        if _HAVE_SDV:
            synth_df = synth.sample(num_rows=int(n_rows))
        else:
            synth_df = synth.sample(num_rows=int(n_rows))
    except TypeError:
        # fallback to different signature
        synth_df = synth.sample(int(n_rows))

    if not isinstance(synth_df, pd.DataFrame):
        synth_df = pd.DataFrame(synth_df)
    return synth_df

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def ks_test_numeric(real: pd.Series, synth: pd.Series) -> float:
    real_clean = real.dropna().astype(float)
    synth_clean = synth.dropna().astype(float)
    if len(real_clean) < 2 or len(synth_clean) < 2:
        return float('nan')
    return float(stats.ks_2samp(real_clean, synth_clean).statistic)


def jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return float('nan')
    p /= p_sum
    q /= q_sum
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.where(a == 0, 1e-12, a)
        b = np.where(b == 0, 1e-12, b)
        return np.sum(a * np.log2(a / b))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def evaluate_synthetic(real: pd.DataFrame, synth: pd.DataFrame, coltypes: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'numeric_ks': {}, 'categorical_js': {}, 'corr_mse': None, 'ml_auc_train_synth_test_real': None}

    for c, t in coltypes.items():
        if c not in real.columns or c not in synth.columns:
            continue
        try:
            if t == 'numerical':
                out['numeric_ks'][c] = ks_test_numeric(real[c], synth[c])
            elif t in ('categorical', 'bool', 'high_cardinality'):
                rcounts = real[c].value_counts(normalize=False)
                scounts = synth[c].value_counts(normalize=False)
                all_idx = sorted(set(rcounts.index).union(set(scounts.index)))
                p = [rcounts.get(i, 0) for i in all_idx]
                q = [scounts.get(i, 0) for i in all_idx]
                out['categorical_js'][c] = jensen_shannon(p, q)
        except Exception:
            # continue even if one column fails
            if t == 'numerical':
                out['numeric_ks'][c] = float('nan')
            else:
                out['categorical_js'][c] = float('nan')

    # correlation matrix mse for numeric columns
    try:
        real_num = real.select_dtypes(include=[np.number])
        synth_num = synth.select_dtypes(include=[np.number])
        common = [c for c in real_num.columns if c in synth_num.columns]
        if len(common) >= 2:
            r_corr = real_num[common].corr().fillna(0).values
            s_corr = synth_num[common].corr().fillna(0).values
            diff = r_corr - s_corr
            out['corr_mse'] = float((diff ** 2).mean())
    except Exception:
        out['corr_mse'] = None

    # Quick ML utility: detect a binary target and compute AUC training on synth, testing on real
    try:
        target = None
        for c in real.columns:
            if real[c].nunique(dropna=True) == 2:
                target = c
                break
        if target is not None:
            Xr = real.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
            yr = real[target].astype(int)
            Xs = synth.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
            ys = synth[target].astype(int)
            common = [c for c in Xr.columns if c in Xs.columns]
            if len(common) >= 1:
                clf = RandomForestClassifier(n_estimators=50, random_state=0)
                clf.fit(Xs[common], ys)
                preds = clf.predict_proba(Xr[common])[:, 1]
                out['ml_auc_train_synth_test_real'] = float(roc_auc_score(yr, preds))
    except Exception:
        out['ml_auc_train_synth_test_real'] = None

    return out

# ---------------------------------------------------------------------------
# Memorization check
# ---------------------------------------------------------------------------

def memorization_check(real: pd.DataFrame, synth: pd.DataFrame, n_neighbors: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {'n_real_rows': len(real), 'n_synth_rows': len(synth)}

    real_num = real.select_dtypes(include=[np.number]).fillna(0)
    synth_num = synth.select_dtypes(include=[np.number]).fillna(0)

    if real_num.shape[1] == 0 or synth_num.shape[1] == 0:
        out.update({'min_dist': None, 'median_dist': None, 'mean_dist': None, 'close_fraction': None})
        return out

    common = [c for c in real_num.columns if c in synth_num.columns]
    if len(common) == 0:
        out.update({'min_dist': None, 'median_dist': None, 'mean_dist': None, 'close_fraction': None})
        return out

    R = real_num[common].values.astype(float)
    S = synth_num[common].values.astype(float)

    mean = R.mean(axis=0, keepdims=True)
    std = R.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Rn = (R - mean) / std
    Sn = (S - mean) / std

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(Rn)
    dists, idxs = nbrs.kneighbors(Sn)
    dists = dists.mean(axis=1) if dists.ndim > 1 else dists

    out['min_dist'] = float(dists.min())
    out['median_dist'] = float(np.median(dists))
    out['mean_dist'] = float(dists.mean())

    thresh = 1e-3
    close_frac = float((dists <= thresh).sum() / len(dists))
    out['close_fraction'] = close_frac
    return out
