import os
import json
import torch
import pickle
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import re
import copy as cp
import random
import math

import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset, random_split
from .sampler import MixtureBatchSampler
from typing import List, Tuple, Dict, Optional, Any, Union

def safe_split_protein_variant(protein_change: str):
    """Parse HGVS-style protein change (e.g. p.R123A) into (prefix, ref, pos, alt).
    Returns None if the string cannot be parsed.
    """
    import re
    if not protein_change:
        return None
    m = re.match(r'^p\.([A-Za-z*?]+)(\d+)([A-Za-z*?=]+)', str(protein_change).strip())
    if not m:
        return None
    return '', m.group(1), int(m.group(2)), m.group(3)
from oncoformer.tokenizer import GeneTokenizer, SPECIALS, RNATokenizer, build_rna_tokenizer, load_rna_vocab

from pydeseq2.dds import DeseqDataSet


# Generic helper: compute a stable permutation for ranking given raw values
def _compute_rank_order(
    rank_by_component: Optional[str],
    rank_ascending: bool,
    rank_values_raw: List[Any],
    token2idx_map: Optional[Dict[str, int]] = None,
) -> Optional[List[int]]:
    """
    Compute a stable argsort order for the given rank values.
    - If a token2idx_map is provided (discrete component), map values to ids and sort by ids.
    - Else attempt numeric sort; if not possible, fallback to string sort.
    Returns a list of indices or None when no ranking should be applied.
    """
    if not rank_by_component or len(rank_values_raw) == 0:
        return None
    s = pd.Series(rank_values_raw)

    # Prefer discrete token-id ordering when available
    if token2idx_map is not None:
        ids = s.map(lambda x: token2idx_map.get(str(x), np.nan))
        if ids.notna().any():
            return ids.sort_values(
                ascending=rank_ascending,
                kind='mergesort',
                na_position='last' if rank_ascending else 'first',
            ).index.tolist()

    # Numeric if possible, else string
    num = pd.to_numeric(s, errors='coerce')
    key = num if num.notna().any() else s.astype(str)
    return key.sort_values(
        ascending=rank_ascending,
        kind='mergesort',
        na_position='last' if rank_ascending else 'first',
    ).index.tolist()

# Utility Functions
def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved object to {filepath}.")

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded object from {filepath}.")
    return obj

def save_tensor(obj, filepath):
    torch.save(obj, filepath)
    print(f"Saved tensor to {filepath}.")

def load_tensor(filepath):
    obj = torch.load(filepath)
    print(f"Loaded tensor from {filepath}.")
    return obj

def get_row_element(row, name, default_value):
    value = row.get(name, default_value)
    if pd.isna(value):
        value = default_value
    return value

def add_file_prefix(path, prefix):
    return os.path.join(
        os.path.dirname(path), prefix + os.path.basename(path)
    )

class OncoformerDataset(Dataset):

   # ------------------ helpers ------------------ #
    def _source_for_modality(self, modality: str, adata: ad.AnnData):
        """Where each modality draws its raw data from (per-modality source)."""
        src = getattr(self, 'mod_source', {}).get(modality, 'anndata')
        if src == 'parquet':
            df = getattr(self, 'parquet_df_by_mod', {}).get(modality, None)
            if df is None:
                raise FileNotFoundError(f"Parquet source for modality '{modality}' not loaded")
            return df
        if src == 'anndata_uns':
            uns_key = getattr(self, 'anndata_uns_keys', {}).get(modality, None)
            if uns_key is None:
                raise KeyError(f"Missing anndata_uns_key for modality '{modality}'")
            return adata.uns[uns_key]
        # default to the whole AnnData
        return adata
    
    def _mod_path(self, modality: str, key: str, default_stub: str) -> Optional[str]:
        """
        Resolve a data path for a modality, adding file prefix. If key missing in config,
        falls back to a sensible default like ./datasets/<stub>.
        """
        md = self.modalities_config[modality]['data']
        val = md.get(key, f'./datasets/{default_stub}')
        return add_file_prefix(val, self.data_file_prefix)
    
    def _mod_handlers(self, modality: str) -> Dict[str, Optional[Any]]:
        """
        Return callables for tokenizer/tokenize/raw/empty per modality (or None).
        Only DNA has a tokenizer & empty-tensor fallback here.
        """
        return {
            'create_tokenizer': getattr(self, f'create_{modality}_tokenizer', None),     # DNA only (for now)
            'tokenize':         getattr(self, f'tokenize_{modality}_data', None),         # DNA & RNA
            'prepare_raw':      getattr(self, f'prepare_{modality}_raw_data', None),      # DNA & RNA
            'empty_tokenized':  getattr(self, f'get_empty_tokenized_{modality}', None),   # DNA only (for now)
        }

    def _resolve_component_atomic_tokens(self, *, vocab: Dict[str, Any], comp_params: Dict[str, Any]) -> set:
        """
        Prefer rows from vocab (atomic_rows). If missing, use tokens from vocab (atomic_tokens).
        Else fall back to comp_params.atomic_tokens. Returns set of TOKEN STRINGS.
        """
        t2i = vocab.get('token_to_idx', {})
        i2t = {int(v): k for k, v in t2i.items()}
    
        rows = vocab.get('atomic_rows', None)
        if rows:
            return {i2t[int(r)] for r in rows if int(r) in i2t}
    
        toks = vocab.get('atomic_tokens', None)
        if toks:
            return set(toks)
    
        toks = comp_params.get('atomic_tokens', None)
        return set(toks) if toks else set()
    
    def _load_precomputed_component_vocabs(self, modality: str):
        """
        Populate:
          self.precomp_token2idx[modality][component] -> dict
          self.precomp_specials[modality][component]  -> dict
          self.component_atomic_tokens[modality][component] -> set(str)
        by reading params.vocab_path for components that supply it (or legacy data.<comp>_vocab_path).
        """
        arch = self.modalities_config.get(modality, {}).get('architecture', {})
        comp_cfg = arch.get('component_configs', {})
        data_root = self.modalities_config.get(modality, {}).get('data', {})
    
        if not hasattr(self, 'precomp_token2idx'):
            self.precomp_token2idx = {}
        if not hasattr(self, 'precomp_specials'):
            self.precomp_specials = {}
        if not hasattr(self, 'component_atomic_tokens'):
            self.component_atomic_tokens = {}
    
        self.precomp_token2idx[modality] = {}
        self.precomp_specials[modality]  = {}
        self.component_atomic_tokens[modality] = {}
    
        for comp_name, cfg in comp_cfg.items():
            params = (cfg.get('params') or {})
            vpath = params.get('vocab_path', None)
            if vpath is None:
                # legacy: allow modalities.<modality>.data.<component>_vocab_path
                vpath = data_root.get(f"{comp_name}_vocab_path", None)
    
            if vpath is None:
                continue  # not a precomputed component
    
            if not os.path.exists(vpath):
                raise FileNotFoundError(f"[{modality}] vocab_path for component '{comp_name}' not found: {vpath}")
    
            with open(vpath, 'r') as f:
                vocab = json.load(f)
    
            tok2idx = vocab.get('token_to_idx', None)
            specials = vocab.get('specials', SPECIALS)
            if not isinstance(tok2idx, dict):
                raise ValueError(f"[{modality}] component '{comp_name}' vocab missing token_to_idx in {vpath}")
    
            self.precomp_token2idx[modality][comp_name] = tok2idx
            self.precomp_specials[modality][comp_name]  = specials
    
            # cache atomic tokens as SET OF STRINGS (from vocab rows/tokens or params override)
            atomic = self._resolve_component_atomic_tokens(vocab=vocab, comp_params=params)
            self.component_atomic_tokens[modality][comp_name] = set(atomic)

    
    def __init__(self, config: Dict[str, Any],
                 load_metadata: bool = False, load_raw: bool = False, clean_start: bool = False, clean_vocab: bool = False):

        self.config = config
        self.modalities_config = self.config['omics']['modalities']
        self.modalities_list = self.config['omics']['architecture'].get('modalities', [])

        assert len(self.modalities_list) > 0,  "At least one modality, eg. 'dna' or 'rna' must be enabled"
        # Legacy compatibility: if load_metadata requested but no metadata config provided, disable metadata loading
        if load_metadata and not bool(self.config.get('metadata', {})):
            load_metadata = False
        
        self.sample_metadata = None
        self.dna_tokenizer = None
        self.tokenized_data = []
        dna_tokenized_data = None
        dna_raw_data = None
        rna_tokenized_data = None
        rna_raw_data = None
        metadata = None

        self.data_file_prefix = self.config.get('data_file_prefix', '')
        self.sample_metadata_path = add_file_prefix(
            self.config.get('sample_metadata_path', './datasets/sample_metadata.pkl'),
            self.data_file_prefix
        )
        self.dna_tokenizer_save_path = None
        self.dna_tokenized_data_path = None
        self.dna_sample_metadata_cols = []
        self.dna_raw_data_path = None
        
        # File paths per modality (tokenizer/tokenized/raw/vocab)
        self.paths = {}
        for m in self.modalities_list:
            vocab_file = (
                (self.modalities_config.get(m, {}) or {})
                .get('vocab', {})
                .get('file', f'./datasets/vocab_{m}.json')
            )
            self.paths[m] = {
                'tokenizer':  self._mod_path(m, 'tokenizer_save_path', f'tokenizer_{m}.pkl'),
                'tokenized':  self._mod_path(m, 'tokenized_data_path', f'tokenized_{m}.pt'),
                'raw':        self._mod_path(m, 'raw_data_path',       f'raw_{m}.pt') if load_raw else None,
                'vocab':      vocab_file,
            }
        
        # Metadata path (single, not per-modality)
        self.metadata_path = None
        if load_metadata:
            self.metadata_path = add_file_prefix(
                self.config['metadata']['data'].get('data_path', './datasets/metadata.pt'),
                self.data_file_prefix
            )
        
        # Other per-modality config hooks that are genuinely special
        self.dna_sample_metadata_cols = []
        if 'dna' in self.modalities_list:
            cols = (
                (self.modalities_config.get('dna', {}) or {})
                .get('data', {})
                .get('anndata_uns_patient_metadata_cols', [])
            )
            if cols is None:
                cols = []
            self.dna_sample_metadata_cols = list(cols)
        self.adata_path = self.config.get('anndata', './datasets/pretraining_data.h5')
        self.dna_anndata_uns_key = self.modalities_config.get('dna', {}).get('data', {}).get(
            'anndata_uns_key', 'paired'
        )
        self.rna_batch_key = self.modalities_config.get('rna', {}).get('data', {}).get('batch_key', None)
        
        # Load vocabs 
        print('Loading vocabs.')
        self.vocabs = {}
        for m in self.modalities_list:
            vpath = self.paths[m]['vocab']
            if not os.path.exists(vpath) or clean_vocab:
                if m == 'rna':
                    data_cfg = (self.modalities_config.get('rna', {}) or {}).get('data', {})
                    n_bins = int(data_cfg.get('discretization', {}).get('n_bins', 32))
                    # Write a stub vocab containing only specials so downstream tokenization
                    # can populate it dynamically on the first pass.
                    self._write_rna_gene_vocab([], n_bins, vpath)
                elif not os.path.exists(vpath):
                    raise FileNotFoundError(f"Expected vocab file for modality '{m}' at {vpath}")
            with open(vpath, 'r') as f:
                self.vocabs[m] = json.load(f)

            self._load_precomputed_component_vocabs(m)
        
        if load_metadata:
            with open(self.config['metadata']['vocab'].get('file', './datasets/vocab_metadata_posttraining.json'), 'r') as f:
                self.vocab_metadata = json.load(f)
        else:
            self.vocab_metadata = None
        
        # Per-modality data sources and optional preloads (strict for parquet)
        self.mod_source = {}
        self.parquet_df_by_mod = {}
        self.anndata_uns_keys = {}
        id_column_map = (self.config.get('sample_index', {}) or {}).get('id_column', {})
        for _m in self.modalities_list:
            mdata = self.modalities_config.get(_m, {}).get('data', {})
            # Default to 'anndata_uns' if anndata_uns_key is provided and source not explicitly set
            src_default = 'anndata_uns' if ('anndata_uns_key' in mdata and mdata.get('anndata_uns_key') is not None) else 'anndata'
            src = mdata.get('source', src_default)
            self.mod_source[_m] = src
            if src == 'parquet':
                pths = mdata.get('parquet_paths', [])
                if isinstance(pths, str):
                    pths = [pths]
                existing = [p for p in pths if isinstance(p, str) and os.path.exists(p)]
                if len(existing) == 0:
                    raise FileNotFoundError(f"{_m}: source=parquet configured but no parquet_paths found on disk")
                parts = [pd.read_parquet(p) for p in existing]
                df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
                id_col = id_column_map.get(_m, 'SampleID')
                if id_col not in df.columns:
                    df = df.copy()
                    df.insert(0, id_col, df.index.astype(str).values)
                self.parquet_df_by_mod[_m] = df
            elif src == 'anndata_uns':
                self.anndata_uns_keys[_m] = mdata.get('anndata_uns_key', None)
            elif src == 'anndata':
                pass
            else:
                raise ValueError(f"Unsupported data.source for modality '{_m}': {src}")

        # Sample metadata (once)
        adata = None
        if os.path.exists(self.sample_metadata_path) and not clean_start:
            print('Loading sample metadata.')
            self.sample_metadata = load_pickle(self.sample_metadata_path)
        else:
            if adata is None:
                print('Loading data.')
                adata = ad.read_h5ad(self.adata_path)
            print('Preparing sample metadata.')
            self.sample_metadata = self.prepare_sample_metadata(adata)
            save_pickle(self.sample_metadata, self.sample_metadata_path)

        self.train_indices, self.val_indices = self._compute_train_val_indices()
        self.train_ids = [str(self.sample_metadata.index[i]) for i in self.train_indices if 0 <= i < len(self.sample_metadata)]
        self.val_ids = [str(self.sample_metadata.index[i]) for i in self.val_indices if 0 <= i < len(self.sample_metadata)]
        
        # Per-modality: load/build tokenizer & tokenized data (and raw if requested)
        self.tokenizers        = {}  # only where present (e.g., dna)
        self.tokenized_by_mod  = {}
        self.raw_by_mod        = {}
        
        for m in self.modalities_list:
            handlers = self._mod_handlers(m)
            p = self.paths[m]
        
            tokenize_fn = handlers['tokenize']
            create_tok  = handlers['create_tokenizer']
            prepare_raw = handlers['prepare_raw']
        
            # Tokenizer + tokenized
            if tokenize_fn is not None:
                need_tokenizer = (create_tok is not None) and (p['tokenizer'] is not None)
        
                load_ok = (
                    (not clean_start) and
                    (not clean_vocab or m != 'rna') and
                    os.path.exists(p['tokenized']) and
                    ((not need_tokenizer) or os.path.exists(p['tokenizer']))
                )
                if load_ok:
                    if need_tokenizer:
                        print(f'Loading {m.upper()} tokenizer.')
                        self.tokenizers[m] = load_pickle(p['tokenizer'])
                    print(f'Loading {m.upper()} tokenized data.')
                    self.tokenized_by_mod[m] = load_tensor(p['tokenized'])
                else:
                    if adata is None:
                        print('Loading data.')
                        adata = ad.read_h5ad(self.adata_path)
                    src = self._source_for_modality(m, adata)
        
                    if need_tokenizer:
                        print(f'Creating {m.upper()} tokenizer.')
                        self.tokenizers[m] = create_tok()
                        save_pickle(self.tokenizers[m], p['tokenizer'])
        
                    print(f'Tokenizing {m.upper()} data.')
                    if m == 'rna':
                        self.tokenized_by_mod[m] = tokenize_fn(src, fit_ids=self.train_ids)
                    else:
                        self.tokenized_by_mod[m] = tokenize_fn(src)
                    save_tensor(self.tokenized_by_mod[m], p['tokenized'])
                    if m == 'rna':
                        rna_tok = self._build_rna_tokenizer()
                        if rna_tok is not None:
                            self.tokenizers['rna'] = rna_tok
                            self.rna_tokenizer = rna_tok

            if m == 'dna' and self.tokenizers.get('dna') is not None:
                self.tokenizers['dna'].ensure_special_tokens(
                    [SPECIALS['pad'], SPECIALS['cls'], SPECIALS['mask'], SPECIALS['unk']]
                )
        
            # Raw (optional)
            if load_raw and (p['raw'] is not None) and (prepare_raw is not None):
                if (not clean_start) and os.path.exists(p['raw']) and (not clean_vocab or m != 'rna'):
                    print(f'Loading {m.upper()} raw data.')
                    self.raw_by_mod[m] = load_tensor(p['raw'])
                else:
                    if adata is None:
                        print('Loading data.')
                        adata = ad.read_h5ad(self.adata_path)
                    src = self._source_for_modality(m, adata)
                    print(f'Preparing {m.upper()} raw data.')
                    if m == 'rna':
                        self.raw_by_mod[m] = prepare_raw(src, fit_ids=self.train_ids)
                    else:
                        self.raw_by_mod[m] = prepare_raw(src)
                    save_tensor(self.raw_by_mod[m], p['raw'])
        
        # Metadata tensors (single cache)
        if load_metadata:
            if (not clean_start) and os.path.exists(self.metadata_path):
                print('Loading metadata.')
                metadata = load_tensor(self.metadata_path)
            else:
                # Determine metadata source AnnData (can differ from RNA counts)
                meta_cfg = self.config.get('metadata', {}).get('data', {})
                meta_source = meta_cfg.get('source', 'anndata')
                meta_path = meta_cfg.get('path', self.adata_path)
                warn_on_mismatch = bool(meta_cfg.get('warn_on_sample_mismatch', False))
                min_overlap = float(meta_cfg.get('min_overlap_fraction', 0.0))

                if meta_source != 'anndata':
                    raise ValueError(f"Unsupported metadata.data.source: {meta_source}")

                if adata is None:
                    print('Loading data.')
                    adata = ad.read_h5ad(self.adata_path)
                if (meta_path is None) or (not os.path.exists(meta_path)):
                    raise FileNotFoundError(f"Metadata AnnData path not found: {meta_path}")
                meta_adata = ad.read_h5ad(meta_path) if (meta_path != self.adata_path) else adata

                # Overlap checks between working sample set and metadata samples
                S = set(map(str, adata.obs.index.tolist()))
                S_meta = set(map(str, meta_adata.obs.index.tolist()))
                inter = S.intersection(S_meta)
                overlap_frac = (len(inter) / max(len(S), 1)) if S else 0.0
                if warn_on_mismatch and (S_meta != S):
                    import warnings
                    warnings.warn(
                        f"metadata.samples ({len(S_meta)}) != working samples ({len(S)}); overlap={len(inter)} (frac={overlap_frac:.3f})",
                        UserWarning,
                    )
                if overlap_frac < min_overlap:
                    raise ValueError(
                        f"Insufficient metadata/sample overlap: {overlap_frac:.3f} < {min_overlap:.3f}"
                    )

                print('Preparing metadata.')
                metadata = self.prepare_metadata(meta_adata)
                save_tensor(metadata, self.metadata_path)
        else:
            metadata = None

        self.rna_tokenizer = self.tokenizers.get('rna', None)
        if 'rna' in self.modalities_list and self.rna_tokenizer is None:
            rna_tok = self._build_rna_tokenizer()
            if rna_tok is not None:
                self.tokenizers['rna'] = rna_tok
                self.rna_tokenizer = rna_tok

        self.dna_tokenizer = self.tokenizers.get('dna', None)

        # Assemble final list of per-sample dicts
        self.tokenized_data = []
        attach_raw = load_raw
        for i, sample_id in enumerate(self.sample_metadata.index.tolist()):
            sample = {'sample_metadata': {'sample_metadata': self.sample_metadata.iloc[[i]]}}
            sample_present: Dict[str, bool] = {}
        
            # attach modality tokenized & raw
            for m in self.modalities_list:
                tok_dict = self.tokenized_by_mod.get(m, None)
                if tok_dict is not None:
                    if sample_id in tok_dict:
                        sample[m] = tok_dict[sample_id]
                        sample_present[m] = True
                    else:
                        # use empty fallback: modality-specific if provided, else generic
                        empty_fn = self._mod_handlers(m)['empty_tokenized']
                        if empty_fn is not None:
                            sample[m] = empty_fn()
                        else:
                            sample[m] = self._get_empty_tokenized_generic(m)
                        sample_present[m] = False
                else:
                    # No tokenized map for modality; synthesize empty if tokenizer exists
                    if m in self.tokenizers:
                        sample[m] = self._get_empty_tokenized_generic(m)
                        sample_present[m] = False
        
                if attach_raw and (m in self.raw_by_mod):
                    raw_map = self.raw_by_mod[m]
                    if sample_id in raw_map:
                        sample[m + '_raw'] = raw_map[sample_id]
        
            # attach metadata tasks
            if metadata is not None:
                sample['sample_metadata'].update(metadata[sample_id])
            # record per-sample modality presence
            sample['present_by_mod'] = sample_present
        
            self.tokenized_data.append(sample)
    
    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def prepare_sample_metadata(self, adata: ad.AnnData):
        adata = adata.copy()
        # Join policy and id column mapping
        join_cfg = self.config.get('sample_index', {}).get('join', 'union')
        id_column_map = (self.config.get('sample_index', {}) or {}).get('id_column', {})

        # Collect per-modality sample id sets
        sample_ids_by_mod = {}
        for _m in self.modalities_list:
            src = self.mod_source.get(_m, 'anndata')
            if src == 'parquet':
                df = self.parquet_df_by_mod.get(_m, None)
                if df is None:
                    sample_ids_by_mod[_m] = set()
                else:
                    id_col = id_column_map.get(_m, 'SampleID')
                    col = df[id_col] if id_col in df.columns else df.index
                    sample_ids_by_mod[_m] = set(map(str, pd.Index(col.astype(str)).unique().tolist()))
            elif src == 'anndata_uns':
                try:
                    uns_key = self.anndata_uns_keys.get(_m, None)
                    df_uns = adata.uns[uns_key] if uns_key is not None else None
                    if isinstance(df_uns, pd.DataFrame):
                        id_col = id_column_map.get(_m, 'SampleID')
                        col = df_uns[id_col] if id_col in df_uns.columns else df_uns.index
                        sample_ids_by_mod[_m] = set(map(str, pd.Index(col.astype(str)).unique().tolist()))
                    else:
                        sample_ids_by_mod[_m] = set()
                except Exception:
                    sample_ids_by_mod[_m] = set()
            else:
                sample_ids_by_mod[_m] = set(map(str, adata.obs.index.astype(str).tolist()))

        # Apply join
        if join_cfg == 'intersection':
            S = None
            for s in sample_ids_by_mod.values():
                S = s if S is None else S.intersection(s)
            S = S or set()
        elif join_cfg in sample_ids_by_mod:
            S = sample_ids_by_mod[join_cfg]
        else:
            S = set()
            for s in sample_ids_by_mod.values():
                S = S.union(s)

        # Build sample metadata frame by reindexing obs to the joined set
        idx = pd.Index(sorted(list(S)))
        sample_metadata = adata.obs.reindex(idx)

        # Optionally append DNA patient metadata columns (only for dna when present in uns)
        if (self.dna_sample_metadata_cols) and (self.mod_source.get('dna', 'anndata_uns') == 'anndata_uns'):
            try:
                dna_sample_metadata = self.prepare_dna_sample_metadata(adata)
                sample_metadata = sample_metadata.join(dna_sample_metadata, how='left')
            except Exception:
                pass

        sample_metadata.index.name = None
        return sample_metadata
    
    def prepare_dna_sample_metadata(self, adata: ad.AnnData):
        dna_sample_metadata = adata.uns[self.dna_anndata_uns_key]
        dna_sample_metadata = dna_sample_metadata.reindex(columns=self.dna_sample_metadata_cols)
        dna_sample_metadata = dna_sample_metadata[~dna_sample_metadata.index.duplicated()]
        return dna_sample_metadata

    def create_dna_tokenizer(self):
        # prefer 'components' but accept legacy 'metadata'
        comp_cfg = self.vocabs['dna'].get('components',
                   self.vocabs['dna'].get('metadata', {}))
        dna_tokenizer = GeneTokenizer(
            component_configs=comp_cfg,
            max_length=self.modalities_config['dna']['architecture'].get('max_length', 128),
            # ensure PAD/CLS/MASK/UNK exist by default
            special_tokens=[SPECIALS['pad'], SPECIALS['cls'], SPECIALS['mask'], SPECIALS['unk']],
        )
        return dna_tokenizer
    
    def tokenize_dna_data(
        self, df: pd.DataFrame,
        patient_id='SampleID',
        gene_id='HugoSymbol',
        mutation_type_id='AlterationInfo',
        protein_change_id='ProteinChange',
        uniprot_column_id='UniprotID',
        read_frac_id='ReadFraction',
        pathogenicity_id='Pathogenicity',
        zygosity_id='ZygosityInfo',
        read_frac_bins=10,
        tokenize=True
    ):
        """
        Build per-sample tensors with aligned components plus any precomputed components
        (e.g., protein/mutation) discovered from config. Order is preserved; sequences
        start with CLS and are padded/truncated to tokenizer.max_length.
    
        Notes on naming:
          - 'alt_type' here is the *component* (discrete label in the tokenizer).
          - 'mutation_atomic_tokens' are special mutation strings (e.g., amplification/deletion)
            that map directly to single mutation tokens (not "uniprot:protein_change").
        """
        tok = self.tokenizers['dna']
        tokenized_data: Dict[str, Dict[str, torch.Tensor]] = {}
    
        # Optional per‑patient ranking controls (non‑backward compatible default)
        dna_data_cfg = self.modalities_config.get('dna', {}).get('data', {})
        rank_by_component = str(dna_data_cfg.get('rank_by_component', 'gene')).lower()
        rank_direction = str(dna_data_cfg.get('rank_direction', 'asc')).lower()
        rank_ascending = (rank_direction == 'asc')

        # Optional baitset restriction (no-op unless 'baitset_filter' is set in config)
        baitset_filter = dna_data_cfg.get('baitset_filter', None)
        if baitset_filter:
            baitset_filter = [str(b) for b in baitset_filter]
            if 'BaitSet' not in df.columns:
                raise KeyError("baitset_filter set but 'BaitSet' column absent from DNA source df")
            before = len(df)
            df = df[df['BaitSet'].astype(str).isin(baitset_filter)]
            print(f'[baitset_filter] BaitSet in {baitset_filter}: {before} -> {len(df)} records')

        # Keep only genes in vocab
        df = df[df[gene_id].isin(tok.component_token2idx['gene'].keys())]
    
        # Precomputed component registries discovered from config (e.g., protein, mutation)
        precomp_tok2idx: Dict[str, Dict[str, int]] = self.precomp_token2idx.get('dna', {})
        precomp_specials: Dict[str, Dict[str, str]] = self.precomp_specials.get('dna', {})
        mutation_atomic = self.component_atomic_tokens.get('dna', {}).get('mutation', set())
        
        # containers to accumulate string tokens for each precomputed component
        precomp_inputs: Dict[str, List[str]] = {name: [] for name in precomp_tok2idx.keys()}
        
        def _pad_seq(seq: List[str], L: int, cls_tok: str, pad_tok: str) -> List[str]:
            seq = [cls_tok] + seq
            return seq[:L] if len(seq) >= L else seq + [pad_tok] * (L - len(seq))

    
        for pid, patient_df in df.groupby(patient_id):
            read_fractions_numeric = pd.to_numeric(patient_df[read_frac_id], errors='coerce')
            max_vaf = read_fractions_numeric.max(skipna=True)
            if max_vaf and max_vaf > 0:
                patient_df = patient_df.copy()
                patient_df['norm_vaf'] = read_fractions_numeric / max_vaf
            else:
                patient_df = patient_df.copy()
                patient_df['norm_vaf'] = None
            # Provide a canonical numeric alias so ranking can reference 'aa_vaf' generically
            patient_df['aa_vaf'] = patient_df['norm_vaf']

            # 1) Single-pass: build component lists and capture a raw rank key per row
            comp = { 'gene': [], 'alt_type': [], 'pathogenicity': [], 'zygosity': [], 'aa_ref': [], 'aa_mut': [], 'aa_pos': [], 'aa_vaf_bin': [] }
            aa_vaf_inputs: List[float] = []
            precomp_inputs: Dict[str, List[str]] = {name: [] for name in precomp_tok2idx.keys()}
            rank_values_raw: List[Any] = []
            max_pos = tok.vocab_sizes.get('aa_pos', 0)
            # resolved once above

            for _, row in patient_df.iterrows():
                gene = get_row_element(row, gene_id, None)
                mutation_type = get_row_element(row, mutation_type_id, None)
                protein_change = str(get_row_element(row, protein_change_id, ''))
                uniprot_id = str(get_row_element(row, uniprot_column_id, ''))
                normalized_vaf = get_row_element(row, 'norm_vaf', None)
                pathogenicity = get_row_element(row, pathogenicity_id, None)
                zygosity = get_row_element(row, zygosity_id, None)

                alt_type = SPECIALS.get('unk', '<unk>')
                aa_ref   = '<na>'
                aa_mut   = '<na>'
                aa_pos   = '<na>'
                aa_vaf_bin = SPECIALS['unk']

                # Treat atomic CN events (e.g., amplification, deletion) as full-prevalence numeric channel
                if mutation_type in mutation_atomic:
                    normalized_vaf = 1.0

                if pd.notna(normalized_vaf):
                    binned_val = int(np.floor(float(normalized_vaf) * read_frac_bins)) + 1
                    binned_val = min(max(binned_val, 1), read_frac_bins)
                    aa_vaf_bin = str(binned_val)

                if mutation_type in ['missense', 'nonsense', 'nonstart', 'nonstop', 'frameshift', 'splice']:
                    split = safe_split_protein_variant(str(protein_change))
                    if split:
                        _, aa_ref_, aa_pos_, aa_mut_ = split
                        aa_pos_ = int(aa_pos_) if isinstance(aa_pos_, (int, np.integer, float)) else None
                        if aa_pos_ is not None and max_pos > 0:
                            aa_pos_ = min(aa_pos_, max_pos - 1)
                        aa_ref = aa_ref_ if aa_ref_ in tok.component_token2idx['aa_ref'] else '<na>'
                        aa_mut_ = str(aa_mut_)
                        aa_mut_ = aa_mut_.replace('fs', '*').replace('ext', '*').replace('?', '*')
                        aa_mut_ = aa_mut_.replace('=', aa_ref)
                        aa_mut = aa_mut_ if aa_mut_ in tok.component_token2idx['aa_mut'] else '<na>'
                        aa_pos = str(aa_pos_) if (aa_pos_ is not None and str(aa_pos_) in tok.component_token2idx['aa_pos']) else '<na>'
                        alt_type = mutation_type
                elif mutation_type in tok.component_token2idx.get('alt_type', {}).keys():
                    alt_type = mutation_type

                comp['gene'].append(gene)
                comp['alt_type'].append(alt_type)
                comp['pathogenicity'].append(pathogenicity)
                comp['zygosity'].append(zygosity)
                comp['aa_ref'].append(aa_ref)
                comp['aa_mut'].append(aa_mut)
                comp['aa_pos'].append(aa_pos)
                comp['aa_vaf_bin'].append(aa_vaf_bin)
                aa_vaf_inputs.append(float(normalized_vaf) if pd.notna(normalized_vaf) else 0.0)

                if 'protein' in precomp_inputs:
                    precomp_inputs['protein'].append(uniprot_id if uniprot_id else SPECIALS['unk'])
                if 'mutation' in precomp_inputs:
                    if (mutation_type in mutation_atomic) and mutation_type:
                        precomp_inputs['mutation'].append(mutation_type)
                    else:
                        precomp_inputs['mutation'].append(
                            f"{uniprot_id}:{protein_change}" if uniprot_id and protein_change else SPECIALS['unk']
                        )

                # capture raw rank key value generically
                if rank_by_component:
                    if rank_by_component in comp:
                        rank_values_raw.append(comp[rank_by_component][-1])
                    elif rank_by_component in precomp_inputs:
                        rank_values_raw.append(precomp_inputs[rank_by_component][-1])
                    else:
                        rank_values_raw.append(get_row_element(row, rank_by_component, None))

            # 2) Optional sort by rank key using generic helper
            if rank_by_component and len(comp['gene']) > 0:
                t2i = tok.component_token2idx.get(rank_by_component, None) or precomp_tok2idx.get(rank_by_component, None)
                order = _compute_rank_order(rank_by_component, rank_ascending, rank_values_raw, token2idx_map=t2i)

                for k in comp:
                    comp[k] = [comp[k][i] for i in order]
                for k in precomp_inputs:
                    precomp_inputs[k] = [precomp_inputs[k][i] for i in order]
                aa_vaf_inputs = [aa_vaf_inputs[i] for i in order]

            encoded = tok.encode(comp, aa_vaf_inputs)
    
            enc_tensors = {
                name: (vals.clone().detach().long() if isinstance(vals, torch.Tensor)
                       else torch.tensor(vals, dtype=torch.long))
                for name, vals in encoded.items() if name != 'aa_vaf'
            }
            if 'aa_vaf' in encoded:
                enc_tensors['aa_vaf'] = torch.tensor(encoded['aa_vaf'], dtype=torch.float)
    
            L = tok.max_length
            for comp_name, str_list in precomp_inputs.items():
                specs = precomp_specials.get(comp_name, SPECIALS)
                cls_tok = specs.get('cls', SPECIALS['cls'])
                pad_tok = specs.get('pad', SPECIALS['pad'])
                unk_tok = specs.get('unk', SPECIALS['unk'])
    
                seq = _pad_seq(str_list, L, cls_tok, pad_tok)
                tok2idx = precomp_tok2idx[comp_name]
                unk_id = tok2idx.get(unk_tok, 0)
                ids = [tok2idx.get(s, unk_id) for s in seq]
                enc_tensors[comp_name] = torch.tensor(ids, dtype=torch.long)
    
            not_empty = (enc_tensors['gene'] != tok.pad_token_id).sum().item() > 1
            if not_empty:
                tokenized_data[pid] = enc_tensors
    
        return tokenized_data



    def prepare_dna_raw_data(self, df: pd.DataFrame,
        gene_id='HugoSymbol'
    ):
        df = df[df[gene_id].isin(self.vocabs['dna']['metadata']['gene'])]
        mut_df = pd.DataFrame({'sample': df.index, 'gene': df[gene_id], 'value': 1})
        mut_df = mut_df.drop_duplicates()
        mut_df = mut_df.pivot(columns='gene', values='value', index='sample')
        mut_df = mut_df.reindex(columns=self.vocabs['dna']['metadata']['gene'])
        mut_df = mut_df.fillna(0)
        mut_matrix = mut_df.to_numpy()
        
        raw_data = {}
        for i, sample_id in enumerate(mut_df.index.tolist()):
            raw_data[sample_id] = {
                'genes': torch.tensor(np.arange(len(self.vocabs['dna']['metadata']['gene']))).long(),
                'values': torch.tensor(mut_matrix[i,:]).long(),
            }
        return raw_data

    # ------------------------------------------------------------------
    # RNA helpers (dynamic HVG vocab + gene mapping)
    # ------------------------------------------------------------------
    def _load_rna_gene_vocab(self) -> Dict[str, int]:
        """Return token->idx map from the HVG vocab JSON if present."""
        vpath = self.paths.get('rna', {}).get('vocab', None)
        tok2idx, _ = load_rna_vocab(vpath)
        return tok2idx

    def _build_rna_tokenizer(self) -> Optional[RNATokenizer]:
        vpath = self.paths.get('rna', {}).get('vocab', None)
        return build_rna_tokenizer(vpath)

    @staticmethod
    def _normalize_rna_kind(value: Any) -> str:
        """Normalize RNA kind inputs to 'bulkrna' or 'scrna'."""
        if value is None:
            return 'bulkrna'

        normalized = str(value).strip().lower()
        allowed = {'bulkrna', 'scrna'}
        if normalized in allowed:
            return normalized

        raise ValueError("Unsupported RNA kind '{value}'. Expected 'bulkrna' or 'scrna'.".format(value=value))

    def _write_rna_gene_vocab(self, gene_list: List[str], n_bins: int, vpath: str):
        """Write token_to_idx mapping including specials to vpath."""
        specials = {'pad': '<pad>', 'cls': '<cls>', 'mask': '<mask>', 'unk': '<unk>'}
        tok2idx = {specials['pad']: 0, specials['cls']: 1, specials['mask']: 2, specials['unk']: 3}
        for g in gene_list:
            if g not in tok2idx:
                tok2idx[g] = len(tok2idx)
        vocab = {
            'token_to_idx': tok2idx,
            'specials': specials,
            'vocab_sizes': {
                'gene': len(tok2idx),
                'expression_bin': int(n_bins)
            }
        }
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        with open(vpath, 'w') as f:
            json.dump(vocab, f)

    def _compute_train_val_indices(self) -> Tuple[List[int], List[int]]:
        """Determine train/val indices according to training config (deterministic)."""
        training_cfg = self.config.get('training', {}) or {}
        seed = int(training_cfg.get('seed', 42))
        n = len(self.sample_metadata.index)
        if n == 0:
            return [], []

        if bool(training_cfg.get('train_test_by_sample_metadata', False)):
            var = training_cfg.get('train_sample_metadata_variable', None)
            train_vals = training_cfg.get('train_sample_metadata_values', [])
            test_vals = training_cfg.get('test_sample_metadata_values', [])
            if var and var in self.sample_metadata.columns:
                series = self.sample_metadata[var]
                train_mask = series.isin(train_vals)
                test_mask = series.isin(test_vals)
                train_idx = [i for i, flag in enumerate(train_mask.tolist()) if flag]
                val_idx = [i for i, flag in enumerate(test_mask.tolist()) if flag]
                if train_idx or val_idx:
                    return train_idx, val_idx

        frac = float(training_cfg.get('train_test_split', 0.8))
        frac = min(max(frac, 0.0), 1.0)
        n_train = int(round(frac * n))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        if n_train <= 0:
            n_train = 0
        train_idx = np.sort(perm[:n_train]).tolist()
        val_idx = np.sort(perm[n_train:]).tolist()
        if not train_idx and n > 0:
            train_idx = [int(perm[0])]
            val_idx = [i for i in range(n) if i != train_idx[0]]
        return train_idx, val_idx

    def tokenize_rna_data(self, adata: ad.AnnData, fit_ids: Optional[List[str]] = None):
        """Comprehensive RNA tokenization supporting bulkrna & scrna with dynamic HVG vocab."""
        cfg_root = self.modalities_config['rna']
        data_cfg = cfg_root['data']
        arch_cfg = cfg_root['architecture']

        kind = self._normalize_rna_kind(data_cfg.get('kind', 'bulkrna'))
        max_len = int(arch_cfg.get('max_length', 1024))
        n_bins = int(data_cfg.get('discretization', {}).get('n_bins', 32))
        zero_bin = bool(data_cfg.get('discretization', {}).get('zero_bin', True))

        rank_by_component_raw = data_cfg.get('rank_by_component', 'expression')
        rank_by_component = str(rank_by_component_raw).strip() if rank_by_component_raw is not None else ''
        rank_component_key = rank_by_component.lower()
        rank_direction = str(data_cfg.get('rank_direction', 'desc')).strip().lower()
        if rank_direction not in ('asc', 'desc'):
            rank_direction = 'desc'
        rank_ascending = (rank_direction == 'asc')

        if fit_ids is None:
            fit_ids = getattr(self, 'train_ids', None)

        vpath = self.paths['rna']['vocab']

        # ---------------- Normalization + transform ----------------
        adata = adata.copy()
        counts_layer = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
        adata.layers['counts'] = counts_layer.copy()
        if kind == 'scrna':
            scrna_norm_cfg = data_cfg.get('scrna_normalization', {})
            target_sum = float(scrna_norm_cfg.get('target_sum', 1e4))
            sc.pp.normalize_total(adata, target_sum=target_sum)

            scrna_transform_cfg = data_cfg.get('scrna_transform', {})
            transform_method = str(scrna_transform_cfg.get('method', 'log1p')).strip().lower()
            if transform_method in ('', 'log1p'):
                sc.pp.log1p(adata)
            else:
                raise NotImplementedError(
                    f"Unsupported scrna transform '{transform_method}'."
                )
        else:  # bulkrna
            # Prepare integer counts for VST
            counts_int = np.rint(counts_layer).clip(min=0).astype(np.int64)
            adata_for_deseq = ad.AnnData(
                counts_int,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            idx_strings = adata_for_deseq.obs.index.astype(str)
            if fit_ids:
                fit_mask = np.isin(idx_strings, np.asarray(fit_ids, dtype=str))
                if not fit_mask.any():
                    fit_mask = np.ones(adata_for_deseq.n_obs, dtype=bool)
            else:
                fit_mask = np.ones(adata_for_deseq.n_obs, dtype=bool)

            adata_fit = adata_for_deseq[fit_mask].copy()
            dds_train = DeseqDataSet(
                adata=adata_fit,
                design='~1',
                quiet=True,
                low_memory=True,
            )
            dds_train.vst_fit(use_design=False)
            vst_counts = dds_train.vst_transform(counts=counts_int).astype(np.float32)
            adata = ad.AnnData(
                vst_counts,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata.layers['counts'] = counts_layer.copy()

        # ---------------- HVG selection ----------------
        gene_tok2idx = self._load_rna_gene_vocab()
        precomputed_genes = [g for g in (gene_tok2idx or {}).keys() if not g.startswith('<')]
        if (
            precomputed_genes
            and len(precomputed_genes) == max_len
            and set(precomputed_genes).issubset(set(map(str, adata.var.index.tolist())))
        ):
            hvg_genes = precomputed_genes
            adata = adata[:, hvg_genes].copy()
        else:
            if kind == 'scrna':
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=max_len,
                    flavor='seurat_v3',
                    layer='counts',
                    batch_key=data_cfg.get('batch_key', None),
                )
                hv_mask = adata.var['highly_variable'].values
                hvg_genes = adata.var.index[hv_mask].tolist()
            else:
                X_vst = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
                idx_strings = adata.obs.index.astype(str)
                if fit_ids:
                    fit_mask = np.isin(idx_strings, np.asarray(fit_ids, dtype=str))
                    if not fit_mask.any():
                        fit_mask = np.ones(adata.n_obs, dtype=bool)
                else:
                    fit_mask = np.ones(adata.n_obs, dtype=bool)
                train_matrix = X_vst[fit_mask, :]
                ddof = 1 if train_matrix.shape[0] > 1 else 0
                variances = np.var(train_matrix, axis=0, ddof=ddof)
                hvg_idx = np.argsort(variances)[-max_len:]
                hvg_genes = adata.var_names[hvg_idx].tolist()
            adata = adata[:, hvg_genes].copy()
            # build & write vocab
            self._write_rna_gene_vocab(hvg_genes, n_bins, vpath)
            gene_tok2idx = self._load_rna_gene_vocab()
            precomputed_genes = hvg_genes

        # ---------------- tensor construction ----------------
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = X.astype(np.float32)
        U = X.copy()
        genes = adata.var.index.values
        L = len(genes)

        pad_id = 0  # as written by _write_rna_gene_vocab

        gene_ids = np.array([gene_tok2idx.get(g, pad_id) for g in genes], dtype=np.int64)

        idx_strings = adata.obs.index.astype(str)
        if fit_ids:
            fit_mask = np.isin(idx_strings, np.asarray(fit_ids, dtype=str))
            if not fit_mask.any():
                fit_mask = np.ones(adata.n_obs, dtype=bool)
        else:
            fit_mask = np.ones(adata.n_obs, dtype=bool)

        fit_matrix = X[fit_mask, :]
        if fit_matrix.shape[0] == 0:
            fit_matrix = X

        nb = n_bins
        if zero_bin:
            nb = max(1, n_bins - 1)

        quantiles = np.linspace(0.0, 1.0, nb + 1)
        q_edges = np.quantile(fit_matrix, quantiles, axis=0)

        def digitize_with_edges(X_sub: np.ndarray, edges: np.ndarray) -> np.ndarray:
            nb_local = edges.shape[0] - 1
            out = np.empty_like(X_sub, dtype=np.int64)
            for j in range(X_sub.shape[1]):
                cut_points = np.maximum.accumulate(edges[1:-1, j])
                out[:, j] = np.digitize(X_sub[:, j], cut_points, right=False)
                out[:, j] = np.clip(out[:, j], 0, nb_local - 1)
            return out

        B = digitize_with_edges(X, q_edges)
        if zero_bin:
            counts_sub = adata.layers.get('counts', None)
            if counts_sub is not None:
                counts_arr = counts_sub.toarray() if hasattr(counts_sub, 'toarray') else np.asarray(counts_sub)
            else:
                counts_arr = np.zeros_like(B, dtype=np.float32)
            B = B + 1
            zero_mask = (counts_arr == 0)
            B[zero_mask] = 0
        B = B.astype(np.int64)

        # geneinfo ids via GenePT vocab
        geneinfo_vocab_path = arch_cfg['component_configs']['geneinfo']['params']['vocab_path']
        with open(geneinfo_vocab_path, 'r') as f:
            gp = json.load(f)
        gp_tok2idx = gp['token_to_idx'] if 'token_to_idx' in gp else {k:int(v) for k,v in gp.items()}
        gp_pad = gp_tok2idx.get('<pad>', 0)
        geneinfo_ids = np.array([gp_tok2idx.get(g, gp_pad) for g in genes], dtype=np.int64)

        out = {}
        allowed_rank_components = {'expression', 'expression_bin', 'gene'}

        for i, sid in enumerate(adata.obs.index.tolist()):
            genes_order = np.array(genes, dtype=str)
            gene_ids_sample = gene_ids.copy()
            geneinfo_ids_sample = geneinfo_ids.copy()
            expression_values = U[i].copy()
            expression_bins = B[i].copy()

            order_indices: Optional[np.ndarray] = None
            key = rank_component_key if rank_component_key in allowed_rank_components else ''
            if key:
                token2idx_map = None
                if key == 'expression':
                    rank_values_raw = expression_values.tolist()
                elif key == 'expression_bin':
                    rank_values_raw = expression_bins.tolist()
                else:  # gene
                    rank_values_raw = genes_order.tolist()
                    token2idx_map = gene_tok2idx

                order = _compute_rank_order(
                    key,
                    rank_ascending,
                    rank_values_raw,
                    token2idx_map=token2idx_map,
                )
                if order is not None:
                    order_indices = np.asarray(order, dtype=np.int64)

            if order_indices is not None:
                gene_ids_sample = gene_ids_sample[order_indices]
                geneinfo_ids_sample = geneinfo_ids_sample[order_indices]
                expression_values = expression_values[order_indices]
                expression_bins = expression_bins[order_indices]

            sample_tensors: Dict[str, torch.Tensor] = {
                'gene': torch.tensor(gene_ids_sample, dtype=torch.long),
                'geneinfo': torch.tensor(geneinfo_ids_sample, dtype=torch.long),
                'expression': torch.tensor(expression_values, dtype=torch.float32),
                'expression_bin': torch.tensor(expression_bins, dtype=torch.long),
            }

            out[str(sid)] = sample_tensors
        return out
    
    
    def prepare_rna_raw_data(self, adata: ad.AnnData, fit_ids: Optional[List[str]] = None):
        """Return per-sample raw float expression on HVGs (post normalization)."""
        cfg = self.modalities_config['rna']
        data_cfg = cfg['data']
        arch_cfg = cfg['architecture']

        kind = self._normalize_rna_kind(data_cfg.get('kind', 'bulkrna'))
        max_len = int(arch_cfg.get('max_length', 1024))

        if fit_ids is None:
            fit_ids = getattr(self, 'train_ids', None)

        adata = adata.copy()
        counts_layer = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
        adata.layers['counts'] = counts_layer.copy()

        # normalization same as tokenize path
        if kind == 'scrna':
            scrna_norm_cfg = data_cfg.get('scrna_normalization', {})
            target_sum = float(scrna_norm_cfg.get('target_sum', 1e4))
            sc.pp.normalize_total(adata, target_sum=target_sum)

            scrna_transform_cfg = data_cfg.get('scrna_transform', {})
            transform_method = str(scrna_transform_cfg.get('method', 'log1p')).strip().lower()
            if transform_method in ('', 'log1p'):
                sc.pp.log1p(adata)
            else:
                raise NotImplementedError(
                    f"Unsupported scrna transform '{transform_method}'."
                )
        else:
            counts_int = np.rint(counts_layer).clip(min=0).astype(np.int64)
            adata_for_deseq = ad.AnnData(
                counts_int,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            idx_strings = adata_for_deseq.obs.index.astype(str)
            if fit_ids:
                fit_mask = np.isin(idx_strings, np.asarray(fit_ids, dtype=str))
                if not fit_mask.any():
                    fit_mask = np.ones(adata_for_deseq.n_obs, dtype=bool)
            else:
                fit_mask = np.ones(adata_for_deseq.n_obs, dtype=bool)

            adata_fit = adata_for_deseq[fit_mask].copy()
            dds_train = DeseqDataSet(
                adata=adata_fit,
                design='~1',
                quiet=True,
                low_memory=True,
            )
            dds_train.vst_fit(use_design=False)
            vst_counts = dds_train.vst_transform(counts=counts_int).astype(np.float32)
            adata = ad.AnnData(
                vst_counts,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            adata.layers['counts'] = counts_layer.copy()

        # HVG reuse or selection
        gene_tok2idx = self._load_rna_gene_vocab()
        if gene_tok2idx and len(gene_tok2idx)-4 == max_len:
            hvg_genes = [g for g in gene_tok2idx.keys() if not g.startswith('<')]
            adata = adata[:, hvg_genes].copy()
        else:
            if kind == 'scrna':
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=max_len,
                    flavor='seurat_v3',
                    layer='counts',
                    batch_key=data_cfg.get('batch_key', None),
                )
                adata = adata[:, adata.var['highly_variable'].values].copy()
            else:
                X_vst = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
                idx_strings = adata.obs.index.astype(str)
                if fit_ids:
                    fit_mask = np.isin(idx_strings, np.asarray(fit_ids, dtype=str))
                    if not fit_mask.any():
                        fit_mask = np.ones(adata.n_obs, dtype=bool)
                else:
                    fit_mask = np.ones(adata.n_obs, dtype=bool)
                train_matrix = X_vst[fit_mask, :]
                ddof = 1 if train_matrix.shape[0] > 1 else 0
                variances = np.var(train_matrix, axis=0, ddof=ddof)
                hvg_idx = np.argsort(variances)[-max_len:]
                hvg_genes = adata.var_names[hvg_idx].tolist()
                adata = adata[:, hvg_genes].copy()

        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = X.astype(np.float32)
        genes = adata.var.index.values
        gene_to_id = {g: i for i, g in enumerate(genes)}

        raw = {}
        for i, sid in enumerate(adata.obs.index.tolist()):
            raw[str(sid)] = {
                'genes': torch.tensor([gene_to_id[g] for g in genes], dtype=torch.long),
                'values': torch.tensor(X[i], dtype=torch.float32),
            }
        return raw



    def prepare_metadata(self, adata: ad.AnnData):
        adata = adata.copy()
        metadata_dict = {}

        for sample_id in adata.obs.index.tolist():
            metadata_dict[sample_id] = {}

        confounder = self.config['metadata']['data'].get('confounder', None)
        if confounder is not None:
            confounder_matrix = adata.obsm[confounder]
            for i, sample_id in enumerate(adata.obs.index.tolist()):
                confounder_tensor = torch.tensor(confounder_matrix[i,:])
                confounder_tensor = confounder_tensor.float()
                metadata_dict[sample_id]['confounder'] = confounder_tensor

        for comp_name in self.config['metadata']['data'].get('discrete', {}).keys():
            metadata_matrix = adata.obsm[comp_name]
            for i, sample_id in enumerate(adata.obs.index.tolist()):
                metadata_tensor = torch.tensor(metadata_matrix[i]).long()
                metadata_dict[sample_id][('discrete', comp_name, 'targets')] = metadata_tensor

        for comp_name in self.config['metadata']['data'].get('point_estimate', {}).keys():
            metadata_matrix = adata.obsm[comp_name]
            for i, sample_id in enumerate(adata.obs.index.tolist()):
                metadata_tensor = torch.tensor(metadata_matrix[i,:]).float()
                metadata_dict[sample_id][('point_estimate', comp_name, 'targets')] = metadata_tensor

        for comp_name in self.config['metadata']['data'].get('series', {}).keys():
            metadata_matrix = adata.obsm[comp_name]
            for i, sample_id in enumerate(adata.obs.index.tolist()):
                metadata_tensor = torch.tensor(metadata_matrix[i,:,:]).float()
                metadata_dict[sample_id][('series', comp_name, 'targets')] = metadata_tensor
                dose_tensor = torch.tensor(adata.uns[comp_name]['doses']).float()
                metadata_dict[sample_id][('series', comp_name, 'doses')] = dose_tensor

        return metadata_dict
            
    def get_empty_tokenized_dna(self):
        """Return an all-PAD sequence for DNA components, including any precomputed components discovered from config."""
        L = int(self.modalities_config['dna']['architecture'].get('max_length', 128))
        empty_data = {}

        # metadata components via tokenizer (gene, alt_type, etc.)
        for comp_name in self.dna_tokenizer.metadata_token2idx.keys():
            seq = [self.dna_tokenizer.cls_token] + [self.dna_tokenizer.pad_token] * (L - 1)
            ids = [self.dna_tokenizer.metadata_token2idx[comp_name][x] for x in seq]
            empty_data[comp_name] = torch.tensor(ids, dtype=torch.long)

        # aa_vaf as float
        vaf = [self.dna_tokenizer.aa_vaf_cls_token] + [self.dna_tokenizer.aa_vaf_pad_token] * (L - 1)
        empty_data['aa_vaf'] = torch.tensor(vaf, dtype=torch.float)

        # any precomputed components (e.g., protein, mutation) discovered from config
        pre = self.precomp_token2idx.get('dna', {})
        pre_specs = self.precomp_specials.get('dna', {})
        for comp_name, tok2idx in pre.items():
            specials = pre_specs.get(comp_name, SPECIALS)
            cls_tok = specials.get('cls', SPECIALS['cls'])
            pad_tok = specials.get('pad', SPECIALS['pad'])
            cls_id  = tok2idx.get(cls_tok, 1)
            pad_id  = tok2idx.get(pad_tok, 0)
            empty_data[comp_name] = torch.tensor([cls_id] + [pad_id]*(L-1), dtype=torch.long)

        return empty_data



# Generic empty fallback for any modality tokenized by tokenizer
    def _get_empty_tokenized_generic(self, modality: str) -> Dict[str, torch.Tensor]:
        tok = self.tokenizers.get(modality, None)
        arch = self.modalities_config.get(modality, {}).get('architecture', {})
        # Prefer seq_length from config; then tokenizer.max_length; then max_length; final fallback 128
        L = int(arch.get('seq_length', getattr(tok, 'max_length', arch.get('max_length', 128))))
        components = list(getattr(tok, 'component_token2idx', {}).keys()) if tok is not None else []
        pad_id = getattr(tok, 'pad_token_id', 0) if tok is not None else 0
        empty_data: Dict[str, torch.Tensor] = {}
        for comp_name in components:
            empty_data[comp_name] = torch.full((L,), pad_id, dtype=torch.long)
        if tok is not None and hasattr(tok, 'float_components') and hasattr(tok, 'component_pad_values'):
            for comp_name in getattr(tok, 'float_components'):
                pad_val = float(tok.component_pad_values.get(comp_name, 0.0))
                empty_data[comp_name] = torch.full((L,), pad_val, dtype=torch.float)
        return empty_data

class OncoformerDataLoader:
    def __init__(self, config: Dict[str, Any],
                 metadata_genes: Optional[List[str]] = None,
                 **kwargs):
        self.config = config
        if metadata_genes is not None:
            assert kwargs.get('load_metadata', False), "'load_metadata' needs to be True if metadata_genes are specified"
        self.dataset = OncoformerDataset(config, **kwargs)
        if metadata_genes is not None:
            self.filter_dataloader_metadata_by_genes(metadata_genes)
        self.train_dataloader = None
        self.val_dataloader = None
        self.create_dataloaders()

    def filter_dataloader_metadata_by_genes(self, metadata_genes):
        tasks = []
        for key in self.dataset.tokenized_data[0]['sample_metadata'].keys():
            if not type(key) is tuple: continue
            if len(key) != 3: continue
            if not key[0] in ('point_estimate', 'series'): continue
            if not key[2] == 'targets': continue
            tasks.append(key[:2])
        for task_type, task_name in tasks:
            gene_idxs = [self.dataset.vocab_metadata[task_name]['features'].index(x) for x in metadata_genes]
            for sample_dict in self.dataset.tokenized_data:
                targets = sample_dict['sample_metadata'][(task_type, task_name, 'targets')]
                targets = [targets.select(0, i) for i in gene_idxs]
                targets = torch.stack(targets, dim=0)
                sample_dict['sample_metadata'][(task_type, task_name, 'targets')] = targets    
        for task_type, task_name in tasks:
            self.dataset.vocab_metadata[task_name]['features'] = metadata_genes
    
    def create_dataloaders(self):
        # Use a fixed random seed for reproducibility
        seed = self.config['training'].get('seed', 42)
        torch.manual_seed(seed)

        dataset_has_split = hasattr(self.dataset, 'train_indices') and hasattr(self.dataset, 'val_indices')
        dataset_train = list(getattr(self.dataset, 'train_indices', []) or [])
        dataset_val = list(getattr(self.dataset, 'val_indices', []) or [])
        total_len = len(self.dataset)

        if dataset_has_split and (dataset_train or dataset_val):
            if not dataset_train and total_len > 0:
                dataset_train = list(range(total_len))
            train_set = set(dataset_train)
            if not dataset_val:
                dataset_val = [i for i in range(total_len) if i not in train_set]
            train_indices = dataset_train
            val_indices = dataset_val
            train_dataset = [self.dataset[i] for i in train_indices]
            val_dataset = [self.dataset[i] for i in val_indices]
        elif self.config['training'].get('train_test_by_sample_metadata', False):
            train_test_variable = self.config['training']['train_sample_metadata_variable']
            train_test_variable = self.dataset.sample_metadata[train_test_variable]
            train_selector = train_test_variable.isin(self.config['training']['train_sample_metadata_values']).tolist()
            test_selector = train_test_variable.isin(self.config['training']['test_sample_metadata_values']).tolist()
            # Build explicit index lists for sampler compatibility
            train_indices = [i for i, is_train in enumerate(train_selector) if is_train]
            val_indices = [i for i, is_val in enumerate(test_selector) if is_val]
            train_dataset = [self.dataset[i] for i in train_indices]
            val_dataset = [self.dataset[i] for i in val_indices]
        else:
            train_test_split = self.config['training'].get('train_test_split', 0.8)
            train_size = int(train_test_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            # Extract indices for sampler compatibility
            try:
                train_indices = list(train_dataset.indices)
                val_indices = list(val_dataset.indices)
            except Exception:
                # Fallback when random_split behavior differs
                train_indices = list(range(train_size))
                val_indices = list(range(train_size, train_size + val_size))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        batch_size = self.config['training'].get('batch_size', 32)
        sampler_cfg = (self.config.get('sampler', {}) or {})
        sampler_enabled = bool(sampler_cfg.get('enabled', True))

        steps_per_epoch: Optional[int] = None

        if sampler_enabled and len(train_indices) > 0:
            # Build mixture batch sampler over the base dataset
            mb_cfg = dict(sampler_cfg)
            mb_cfg['batch_size'] = batch_size
            # Default cohorts to matched:1.0 if not provided
            cohorts_cfg = mb_cfg.get('cohorts', None)
            if not cohorts_cfg:
                mb_cfg['cohorts'] = {'matched': 1.0}
            train_batch_sampler = MixtureBatchSampler(
                dataset=self.dataset,
                sampler_cfg=mb_cfg,
                indices=train_indices,
                rng=random.Random(seed),
            )

            self.train_dataloader = DataLoader(
                self.dataset,
                batch_sampler=train_batch_sampler,
                collate_fn=self.collate_fn,
            )
            steps_per_epoch = len(train_batch_sampler)
        else:
            # Fallback: vanilla DataLoader on subset/list
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            train_size = max(len(train_dataset), 0)
            steps_per_epoch = int(max(math.ceil(max(train_size, 1) / max(batch_size, 1)), 1))

        # Validation loader: default sequential batches over val split
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        # Update training schedule metadata
        training_cfg = self.config.setdefault('training', {})
        steps_per_epoch = int(max(steps_per_epoch or 0, 1))
        training_cfg['training_len'] = steps_per_epoch

        warmup_fraction = training_cfg.get('scheduler_warmup_fraction', None)
        if warmup_fraction is None:
            warmup_fraction = 0.1
        warmup_fraction = float(warmup_fraction)
        training_cfg['scheduler_warmup_fraction'] = warmup_fraction

        num_epochs = int(training_cfg.get('num_epochs', 20))
        warmup_steps = int(math.floor(steps_per_epoch * num_epochs * warmup_fraction))
        training_cfg['scheduler_warmup_steps'] = warmup_steps

        # Auto-compute class weights if requested (training-time helper)
        try:
            cw_cfg = self.config.get('metadata', {}).get('class_weighting', {})
            enabled = bool(cw_cfg.get('enabled', False))
            compute_if_missing = bool(cw_cfg.get('compute_if_missing', False))
            if enabled and compute_if_missing and (self.train_dataloader is not None):
                disc_cfg = self.config.get('metadata', {}).get('data', {}).get('discrete', {})
                apply_to = cw_cfg.get('apply_to', 'all')
                task_list = list(disc_cfg.keys()) if apply_to == 'all' else list(apply_to)
                # Only compute if at least one target task lacks weights
                def _missing_weights(t):
                    tc = disc_cfg.get(t, {})
                    return (not isinstance(tc.get('class_weights', None), (list, tuple))) or (len(tc.get('class_weights', [])) == 0)
                if any(_missing_weights(t) for t in task_list):
                    self.compute_and_update_class_weights(
                        self.config,
                        self.train_dataloader,
                        tasks=apply_to,
                        method=cw_cfg.get('method', 'effective'),
                        beta=float(cw_cfg.get('beta', 0.999)),
                        normalize=cw_cfg.get('normalize', 'mean'),
                        exclude_ignored=True,
                    )
        except Exception:
            # silent no-op if class-weight computation is not applicable
            pass

    def collate_fn(self, batch):
        """
        Collate function to combine samples into a forward-only omics_format batch:
          - 'omics_inputs': per-modality dicts of stacked component tensors
          - 'omics_masks': per-modality attention masks [B,L]
          - 'sample_metadata': stacked metadata dict (DataFrame and tensors)
        """
        B = len(batch)
        # 1) Build omics_inputs by stacking per component for each modality
        omics_inputs: Dict[str, Dict[str, Any]] = {}
        present_flags_by_mod: Dict[str, List[bool]] = {m: [] for m in self.dataset.modalities_list}
        for m in self.dataset.modalities_list:
            # components present for modality m in first sample
            comps = list(batch[0].get(m, {}).keys())
            omics_inputs[m] = {c: [] for c in comps}
            for sample in batch:
                for c in comps:
                    omics_inputs[m][c].append(sample[m][c])
                present_flags_by_mod[m].append(bool(sample.get('present_by_mod', {}).get(m, True)))
            # stack components
            for c in comps:
                first = omics_inputs[m][c][0]
                if isinstance(first, torch.Tensor):
                    omics_inputs[m][c] = torch.stack(omics_inputs[m][c], dim=0)
                elif isinstance(first, pd.DataFrame):
                    omics_inputs[m][c] = pd.concat(omics_inputs[m][c], axis=0)
                elif isinstance(first, np.ndarray):
                    omics_inputs[m][c] = np.stack(omics_inputs[m][c], axis=0)
                else:
                    # leave lists of python objects as-is
                    pass

        # 2) Build omics_masks per modality using tokenizer guidance
        omics_masks: Dict[str, torch.Tensor] = {}
        for m in self.dataset.modalities_list:
            tok = self.dataset.tokenizers.get(m, None)
            # choose mask component
            mask_comp = getattr(tok, 'mask_component', None)
            if not mask_comp or (mask_comp not in omics_inputs[m]):
                # fallback: first integer tensor component [B,L]
                mask_comp = None
                for c, v in omics_inputs[m].items():
                    if isinstance(v, torch.Tensor) and v.dtype in (torch.long, torch.int64, torch.int32):
                        if v.ndim == 2:
                            mask_comp = c
                            break
            if mask_comp is None:
                # last resort: skip mask (all zeros)
                L = next((v.shape[1] for v in omics_inputs[m].values() if isinstance(v, torch.Tensor) and v.ndim >= 2), 0)
                omics_masks[m] = torch.zeros(B, L, dtype=torch.long)
                continue
            pad_id = getattr(tok, 'pad_token_id', 0)
            v = omics_inputs[m][mask_comp]
            mask = (v != pad_id).long() if isinstance(v, torch.Tensor) else torch.zeros(B, 0, dtype=torch.long)
            # Zero out rows where modality was not present for the sample
            present_vec = torch.tensor(present_flags_by_mod[m], dtype=torch.bool, device=mask.device)
            if mask.numel() > 0:
                mask[~present_vec] = 0
            omics_masks[m] = mask

        # 3) Collate sample_metadata (stack DataFrame/tensors using existing logic)
        meta_inputs = {}
        top_key = 'sample_metadata'
        component_names = batch[0][top_key].keys() if isinstance(batch[0][top_key], dict) else []
        if component_names:
            inputs = {comp_name: [] for comp_name in component_names}
            for sample in batch:
                for comp_name in component_names:
                    inputs[comp_name].append(sample[top_key][comp_name])
            for comp_name in component_names:
                first = inputs[comp_name][0]
                if isinstance(first, torch.Tensor):
                    inputs[comp_name] = torch.stack(inputs[comp_name], dim=0)
                elif isinstance(first, pd.DataFrame):
                    inputs[comp_name] = pd.concat(inputs[comp_name], axis=0)
                elif isinstance(first, np.ndarray):
                    inputs[comp_name] = np.stack(inputs[comp_name], axis=0)
            meta_inputs = inputs
        else:
            meta_inputs = [s[top_key] for s in batch]

        return { 'omics_inputs': omics_inputs, 'omics_masks': omics_masks, 'sample_metadata': meta_inputs }

    # ----------------------------
    # Training-time helper: class weights
    # ----------------------------
    def _task_ignore_ids_from_config(self, task_name: str) -> List[int]:
        """
        Return label ids to ignore for a discrete task based on config ignore_patterns
        and the loaded metadata vocab.
        """
        ignore_ids: List[int] = []
        try:
            meta_cfg = self.config.get('metadata', {})
            data_cfg = meta_cfg.get('data', {}).get('discrete', {})
            ignore_patterns = [str(x).lower() for x in data_cfg.get(task_name, {}).get('ignore_patterns', [])]
            if not ignore_patterns:
                return ignore_ids
            vocab_levels = self.dataset.vocab_metadata[task_name]['levels'] if self.dataset.vocab_metadata else None
            if vocab_levels is None:
                return ignore_ids
            for i, lbl in enumerate(vocab_levels):
                ln = str(lbl).lower()
                if any(pat in ln for pat in ignore_patterns):
                    ignore_ids.append(int(i))
        except Exception:
            pass
        return ignore_ids

    def compute_and_update_class_weights(
        self,
        config: Dict[str, Any],
        train_iterable: Any,
        *,
        tasks: Optional[Union[List[str], str]] = None,
        method: str = "effective",
        beta: float = 0.999,
        normalize: str = "mean",
        exclude_ignored: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-class weights for discrete metadata tasks from the training split and
        write them into the provided config at metadata.data.discrete.<task>.class_weights.

        - method: "effective" (Cui et al.) or "inverse"
        - normalize: "mean" to scale weights to mean 1.0, or "none"
        - tasks: list of task names or "all" (default: all discrete tasks in config)
        - exclude_ignored: if True, excluded label ids (from ignore_patterns) are not counted
        """
        from collections import Counter

        # Resolve target tasks
        disc_cfg = config.get('metadata', {}).get('data', {}).get('discrete', {})
        if tasks is None or tasks == 'all':
            task_list = list(disc_cfg.keys())
        else:
            task_list = list(tasks)

        # Helper to convert counts -> weights
        def _weights_from_counts(counts_vec: torch.Tensor) -> torch.Tensor:
            n = counts_vec.clamp(min=1).float()
            if method.lower() == 'inverse':
                w = 1.0 / n
            else:
                # effective number of samples
                b = float(beta)
                w = (1.0 - b) / (1.0 - torch.pow(b, n))
            if normalize.lower() == 'mean':
                w = w / w.mean()
            return w

        weights_by_task: Dict[str, torch.Tensor] = {}

        # Precompute ignore ids per task (optional)
        ignore_map: Dict[str, List[int]] = {}
        if exclude_ignored:
            for t in task_list:
                ignore_map[t] = self._task_ignore_ids_from_config(t)

        # Accumulate counts by iterating once over the training iterable
        counters: Dict[str, Counter] = {t: Counter() for t in task_list}
        for batch in train_iterable:
            for t in task_list:
                key = ('discrete', t, 'targets')
                if 'sample_metadata' not in batch or key not in batch['sample_metadata']:
                    continue
                labels = batch['sample_metadata'][key].view(-1).detach().cpu().tolist()
                if exclude_ignored and t in ignore_map and ignore_map[t]:
                    ignored = set(ignore_map[t])
                    labels = [x for x in labels if x not in ignored]
                counters[t].update(labels)

        # Convert to dense tensors and write to config
        for t in task_list:
            # Determine number of classes from vocab
            try:
                levels = self.dataset.vocab_metadata[t]['levels']
                nc = int(len(levels))
            except Exception:
                nc = max(list(counters[t].keys()) + [0]) + 1 if counters[t] else 0
            if nc <= 0:
                continue
            cnt = torch.tensor([counters[t].get(i, 0) for i in range(nc)], dtype=torch.float)
            w = _weights_from_counts(cnt)
            weights_by_task[t] = w
            # Persist into config as plain floats list
            disc_cfg.setdefault(t, {})['class_weights'] = [float(x) for x in w.tolist()]
            # Optional audit info
            disc_cfg[t]['class_weighting_summary'] = {
                'counts': [int(x) for x in cnt.tolist()],
                'method': method,
                'beta': float(beta),
                'normalize': normalize,
            }

        return weights_by_task

