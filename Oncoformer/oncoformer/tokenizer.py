import os
import json
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple, Iterable

SPECIALS = {'pad': '<pad>', 'cls': '<cls>', 'mask': '<mask>', 'unk': '<unk>'}


def load_rna_vocab(vocab_path: Optional[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load an HVG vocab JSON and return (token_to_idx, specials)."""

    specials = dict(SPECIALS)
    if not vocab_path or not os.path.exists(vocab_path):
        return {}, specials

    try:
        with open(vocab_path, 'r') as f:
            payload = json.load(f)
    except Exception:
        return {}, specials

    specials = payload.get('specials', specials)

    def _ensure_specials(mapping: Dict[str, int]) -> Dict[str, int]:
        mapping = dict(mapping)
        next_idx = max(mapping.values(), default=-1) + 1
        for name in ('pad', 'cls', 'mask', 'unk'):
            tok = specials.get(name, SPECIALS[name])
            if tok not in mapping:
                mapping[tok] = next_idx
                next_idx += 1
        return mapping

    if 'token_to_idx' in payload and isinstance(payload['token_to_idx'], dict):
        tok2idx = {str(k): int(v) for k, v in payload['token_to_idx'].items()}
        tok2idx = _ensure_specials(tok2idx)
        return tok2idx, specials

    if 'genes' in payload and isinstance(payload['genes'], list):
        tok2idx: Dict[str, int] = {}
        for name in ('pad', 'cls', 'mask', 'unk'):
            tok = specials.get(name, SPECIALS[name])
            if tok not in tok2idx:
                tok2idx[tok] = len(tok2idx)
        for gene in payload['genes']:
            tok = str(gene)
            if tok not in tok2idx:
                tok2idx[tok] = len(tok2idx)
        return tok2idx, specials

    # fallback: treat payload as token->index mapping
    tok2idx = {str(k): int(v) for k, v in payload.items() if isinstance(v, (int, float))}
    tok2idx = _ensure_specials(tok2idx)
    return tok2idx, specials


class RNATokenizer:
    __slots__ = (
        'token_to_idx',
        'pad_token_id',
        'cls_token_id',
        'mask_token_id',
        'component_token2idx',
        'metadata_idx2token',
    )

    def __init__(self, token_to_idx: Dict[str, int], specials: Dict[str, str]):
        self.token_to_idx = token_to_idx
        pad_tok = specials.get('pad', SPECIALS['pad'])
        cls_tok = specials.get('cls', SPECIALS['cls'])
        mask_tok = specials.get('mask', SPECIALS['mask'])

        self.pad_token_id = int(token_to_idx.get(pad_tok, 0))
        self.cls_token_id = int(token_to_idx.get(cls_tok, 1))
        self.mask_token_id = int(token_to_idx.get(mask_tok, self.pad_token_id))

        self.component_token2idx = {'gene': token_to_idx}
        self.metadata_idx2token = {
            'gene': {int(idx): tok for tok, idx in token_to_idx.items()}
        }


def build_rna_tokenizer(vocab_path: Optional[str]) -> Optional[RNATokenizer]:
    tok2idx, specials = load_rna_vocab(vocab_path)
    if not tok2idx:
        return None
    return RNATokenizer(tok2idx, specials)

def _stack_rows(rows):
    if not rows:
        raise ValueError("No rows to stack")
    return torch.cat(rows, dim=0)

# Handle precomputed embeddings
    
def build_embedding_assets_from_raw(
    raw_path: str,
    out_weight_path: str,
    out_vocab_path: str,
    *,
    filter_fn=None,                  # e.g., protein: lambda k: ':' not in k
    extra_tokens: Optional[List[str]] = None,  
    specials: Dict[str, str] = SPECIALS,
    dtype: Union[str, int, torch.dtype] = "bf16", 
) -> Tuple[str, Dict[str, Any]]:
    """
    raw_path: torch.load()-able mapping {str: FloatTensor[D]} (or {"weight":..., "ids":[...]})
    Writes weight.pt ([V,D]) and vocab.json. Returns (out_weight_path, vocab_dict)
    """

    def _resolve_dtype(d) -> torch.dtype:
        if isinstance(d, torch.dtype):
            return d
        s = str(d).lower()
        if s in ("fp16", "float16", "half", "16"):
            return torch.float16
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp32", "float32", "32"):
            return torch.float32
        return torch.float16  # default

    target_dtype = _resolve_dtype(dtype)

    raw = torch.load(raw_path, map_location='cpu')
    token_vec_pairs = []

    # Accept either dict[token]->vec or dict with {"weight": [V,D], "ids":[...]}
    if isinstance(raw, dict) and 'weight' in raw and ('ids' in raw or 'id2idx' in raw):
        W = raw['weight'].float()
        if 'ids' in raw:
            ids = list(raw['ids'])
        else:
            id2idx = dict(raw['id2idx'])
            ids = [None] * W.size(0)
            for k, i in id2idx.items():
                if 0 <= i < len(ids):
                    ids[i] = k
            for i in range(len(ids)):
                if ids[i] is None:
                    ids[i] = f"tok_{i}"
        for i, tok in enumerate(ids):
            if (filter_fn is None) or filter_fn(tok):
                token_vec_pairs.append((tok, W[i].unsqueeze(0)))
        D = W.size(1)
    elif isinstance(raw, dict):
        # token -> tensor
        for tok, vec in raw.items():
            if not torch.is_tensor(vec):
                continue
            if (filter_fn is None) or filter_fn(tok):
                token_vec_pairs.append((tok, vec.float().view(1, -1)))
        if not token_vec_pairs:
            raise ValueError(f"No tokens survived filter for {raw_path}")
        D = token_vec_pairs[0][1].size(2) if token_vec_pairs[0][1].dim()==3 else token_vec_pairs[0][1].size(1)
    else:
        raise ValueError(f"Unsupported raw format in {raw_path}")

    # Assemble special rows (zeros) first for consistent indices
    tok2idx: Dict[str, int] = {}
    rows: List[torch.Tensor] = []
    for name in ('pad', 'cls', 'mask', 'unk'):
        tok = specials[name]
        tok2idx[tok] = len(tok2idx)
        rows.append(torch.zeros(1, D))  # keep as fp32; cast once at the end

    # Append precomputed tokens
    for tok, row in token_vec_pairs:
        if tok in tok2idx:
            continue
        tok2idx[tok] = len(tok2idx)
        rows.append(row)

    # Append extra tokens (zeros), record their row indices
    atomic_rows: List[int] = []
    if extra_tokens:
        for tok in extra_tokens:
            if tok not in tok2idx:
                tok2idx[tok] = len(tok2idx)
                rows.append(torch.zeros(1, D))
            atomic_rows.append(tok2idx[tok])

    W_final = _stack_rows(rows)  # [V, D]
    # NEW: cast once, right before saving
    if W_final.dtype != target_dtype:
        W_final = W_final.to(target_dtype)
    torch.save(W_final, out_weight_path)

    vocab = {
        "token_to_idx": tok2idx,
        "specials": specials,
        "embedding_dim": int(D),
    }

    # Forward-only atomic metadata
    if extra_tokens:
        vocab["atomic_tokens"] = list(extra_tokens)
        vocab["atomic_rows"] = [tok2idx[tok] for tok in extra_tokens if tok in tok2idx]

    with open(out_vocab_path, 'w') as f:
        json.dump(vocab, f)

    return out_weight_path, vocab


def ensure_embedding_assets(
    *,
    name: str,                    # for default filenames
    raw_path: Optional[str],
    weight_path: Optional[str],
    vocab_path: Optional[str],
    filter_fn=None,
    extra_tokens: Optional[List[str]] = None,
    specials: Dict[str, str] = SPECIALS,
    dtype: Union[str, int, torch.dtype] = "fp16", 
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (weight_path, vocab_path, vocab_dict). Builds from raw if needed.
    If weight_path & vocab_path exist, they are loaded and returned.
    If only raw_path is given, generates files next to raw_path with suffixes.
    """
    if weight_path and vocab_path and os.path.exists(weight_path) and os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        return weight_path, vocab_path, vocab

    if raw_path is None:
        raise ValueError(f"Embedding assets for {name}: provide either existing weight/vocab or a raw_path")

    base = os.path.splitext(raw_path)[0]
    if weight_path is None:
        weight_path = f"{base}.{name}.weight.pt"
    if vocab_path is None:
        vocab_path = f"{base}.{name}.vocab.json"

    weight_path, vocab = build_embedding_assets_from_raw(
        raw_path,
        weight_path,
        vocab_path,
        filter_fn=filter_fn,
        extra_tokens=extra_tokens,
        specials=specials,
        dtype=dtype,  # NEW
    )
    return weight_path, vocab_path, vocab


class GeneTokenizer:
    """
    Order-preserving, PLM-agnostic tokenizer for DNA components.
    - Uses 'components' naming (but exposes 'metadata_*' aliases for back-compat).
    - Only includes special tokens you actually pass in (default: PAD + CLS).
    - No hidden sorting; truncates/pads to max_length.
    - Supports optional float channel 'aa_vaf'.
    """
    def __init__(
        self,
        component_configs: Dict[str, List[str]],
        max_length: int = 128,
        use_cls_token: bool = True,
        special_tokens: Optional[List[str]] = None,  # default to only PAD + CLS
        pad_token:  str = "<pad>",
        cls_token:  str = "<cls>",
        # Optional extra specials if you *really* want them; otherwise omit.
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = None,
    ):
        # --- specials (default to only PAD/CLS unless provided) ---
        if special_tokens is None:
            special_tokens = [pad_token, cls_token]
        else:
            # ensure PAD/CLS exist
            need = {pad_token, cls_token}
            for t in need:
                if t not in special_tokens:
                    special_tokens.append(t)

        self.use_cls_token = bool(use_cls_token)
        self.special_tokens = list(special_tokens)
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.unk_token = unk_token if (unk_token in self.special_tokens) else None
        self.mask_token = mask_token if (mask_token in self.special_tokens) else None

        # numeric channel defaults
        self.aa_vaf_pad_token = 0.0
        self.aa_vaf_cls_token = 1.0

        # Build vocabularies for components
        self.component_configs = component_configs or {}
        self.component_token2idx: Dict[str, Dict[str,int]] = {}
        self.component_idx2token: Dict[str, Dict[int,str]] = {}
        self.component_vocab_sizes: Dict[str, int] = {}

        for comp_name, tokens in self.component_configs.items():
            tokens_with_specials = self.special_tokens + list(tokens)
            token2idx = {tok: i for i, tok in enumerate(tokens_with_specials)}
            idx2token = {i: tok for tok, i in token2idx.items()}
            self.component_token2idx[comp_name] = token2idx
            self.component_idx2token[comp_name] = idx2token
            self.component_vocab_sizes[comp_name] = len(token2idx)

        # IDs copied from the 'gene' component (required)
        if 'gene' not in self.component_token2idx:
            raise ValueError("GeneTokenizer requires a 'gene' component in component_configs.")
        gene_map = self.component_token2idx['gene']
        self.pad_token_id = gene_map[self.pad_token]
        self.cls_token_id = gene_map[self.cls_token]
        self.unk_token_id = gene_map.get(self.unk_token, self.pad_token_id)  # fallback to PAD if UNK not used
        self.mask_token_id = gene_map.get(self.mask_token, self.pad_token_id)
        # Preferred component to derive attention masks from
        self.mask_component: str = 'gene'

        self.max_length = int(max_length)
        self.vocab_sizes = dict(self.component_vocab_sizes)  # alias for caller expectations

        # --------- Back-compat aliases (so old code still works) ---------
        self.metadata_components = self.component_configs
        self.metadata_token2idx = self.component_token2idx
        self.metadata_idx2token = self.component_idx2token
        self.metadata_vocab_sizes = self.component_vocab_sizes
        self.na_token = None
        self.special_tokens_id = [self.pad_token_id, self.cls_token_id]
        self.filter_tokens_id  = [self.pad_token_id, self.cls_token_id]

        # Component typing/pad values for modality-agnostic handling
        # Expose float components and their pad values (if any)
        self.float_components = []
        self.component_pad_values: Dict[str, float] = {}
        # GeneTokenizer supports optional 'aa_vaf' float channel
        self.float_components.append('aa_vaf')
        self.component_pad_values['aa_vaf'] = float(self.aa_vaf_pad_token)

    def tokenize(
        self,
        component_sequences: Dict[str, List[str]],
        aa_vaf_sequence: Optional[List[float]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Build token lists per component. Order is preserved.
        Adds leading CLS if use_cls_token. Truncates/pads to max_length.
        """
        num_items = len(component_sequences['gene'])
        tokens: Dict[str, List[Any]] = {}

        # discrete components
        for comp_name in self.component_configs.keys():
            seq = [self.cls_token] if self.use_cls_token else []
            if comp_name in component_sequences:
                comp_seq = component_sequences[comp_name]
                if len(comp_seq) != num_items:
                    raise ValueError(f"Component '{comp_name}' length ({len(comp_seq)}) "
                                     f"does not match 'gene' length ({num_items}).")
                seq += comp_seq
            else:
                # if component not provided, fill with PAD
                seq += [self.pad_token] * num_items
            tokens[comp_name] = seq

        # float channel (optional)
        if aa_vaf_sequence is not None:
            seq = [self.aa_vaf_cls_token] if self.use_cls_token else []
            seq += list(aa_vaf_sequence)
            tokens['aa_vaf'] = seq

        # truncate / pad to max_length
        for key, seq in list(tokens.items()):
            if len(seq) >= self.max_length:
                tokens[key] = seq[:self.max_length]
            else:
                pad_fill = self.pad_token if key != 'aa_vaf' else self.aa_vaf_pad_token
                tokens[key] = seq + [pad_fill] * (self.max_length - len(seq))

        return tokens

    def numericalize(self, tokens: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        indices: Dict[str, torch.Tensor] = {}
        for comp_name in self.component_configs.keys():
            tok2idx = self.component_token2idx[comp_name]
            if self.unk_token is not None and self.unk_token in tok2idx:
                unk_id = tok2idx[self.unk_token]
            else:
                unk_id = tok2idx[self.pad_token]  # safe fallback
            seq = [tok2idx.get(str(t), unk_id) for t in tokens[comp_name]]
            indices[comp_name] = torch.tensor(seq, dtype=torch.long)
        if 'aa_vaf' in tokens:
            indices['aa_vaf'] = torch.tensor(tokens['aa_vaf'], dtype=torch.float)
        return indices

    def encode(
        self,
        component_sequences: Dict[str, List[str]],
        aa_vaf_sequence: Optional[List[float]] = None,
        as_tensor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tokens = self.tokenize(component_sequences, aa_vaf_sequence)
        return self.numericalize(tokens)

    def decode(self, indices: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for comp_name in self.component_configs.keys():
            idx2tok = self.component_idx2token[comp_name]
            out[comp_name] = [idx2tok.get(int(i), self.pad_token) for i in indices[comp_name]]
        if 'aa_vaf' in indices:
            out['aa_vaf'] = indices['aa_vaf'].tolist()
        return out

    def ensure_special_tokens(self, required_tokens: Iterable[str]) -> None:
        """
        Ensure that each required special token exists across all component vocabularies.
        Adds missing tokens to component mappings and updates cached ids.
        """
        required = list(required_tokens)
        # Ensure internal special token list contains all required tokens (preserve order, avoid duplicates)
        for tok in required:
            if tok not in self.special_tokens:
                self.special_tokens.append(tok)

        for tok in required:
            for comp_name, token2idx in self.component_token2idx.items():
                if tok not in token2idx:
                    idx = len(token2idx)
                    token2idx[tok] = idx
                    self.component_idx2token[comp_name][idx] = tok
                    self.component_vocab_sizes[comp_name] = len(token2idx)

        # Refresh cached ids for key specials if now present
        pad_tok = self.pad_token if self.pad_token in self.special_tokens else SPECIALS['pad']
        cls_tok = self.cls_token if self.cls_token in self.special_tokens else SPECIALS['cls']
        mask_tok = SPECIALS['mask']
        unk_tok = SPECIALS['unk']

        if pad_tok in self.component_token2idx['gene']:
            self.pad_token = pad_tok
            self.pad_token_id = self.component_token2idx['gene'][pad_tok]
        if cls_tok in self.component_token2idx['gene']:
            self.cls_token = cls_tok
            self.cls_token_id = self.component_token2idx['gene'][cls_tok]

        if mask_tok in self.component_token2idx['gene']:
            self.mask_token = mask_tok
            self.mask_token_id = self.component_token2idx['gene'][mask_tok]
        else:
            self.mask_token = None
            self.mask_token_id = self.pad_token_id

        if unk_tok in self.component_token2idx['gene']:
            self.unk_token = unk_tok
            self.unk_token_id = self.component_token2idx['gene'][unk_tok]
        else:
            self.unk_token = None
            self.unk_token_id = self.pad_token_id

        # Refresh cached lists
        gene_map = self.component_token2idx['gene']
        self.special_tokens_id = [gene_map[tok] for tok in self.special_tokens if tok in gene_map]
        self.filter_tokens_id = [self.pad_token_id, self.cls_token_id]
    