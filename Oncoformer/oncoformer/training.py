import numpy as np
import copy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional, Any
from torch import Tensor

import math

from oncoformer.tokenizer import GeneTokenizer

def _build_keep_mask(attn_mask: torch.Tensor, keep_ratio: float, keep_cls: bool = True) -> torch.Tensor:
    """
    Downsample a binary attention mask: keep ~keep_ratio of the VALID tokens.
    Always keeps CLS at index 0 if keep_cls=True.
    Shapes: attn_mask [B, L] -> keep_mask [B, L] (bool)
    """
    B, L = attn_mask.shape
    keep = torch.zeros_like(attn_mask, dtype=torch.bool)
    valid = attn_mask.bool()
    for i in range(B):
        idx = torch.nonzero(valid[i], as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if keep_cls:
            keep[i, 0] = True if valid[i, 0] else False
            idx = idx[idx != 0]
        k = max(1, int(round(keep_ratio * idx.numel())))
        if k > 0:
            pick = idx[torch.randperm(idx.numel(), device=attn_mask.device)[:k]]
            keep[i, pick] = True
    return keep

def _maybe_moddrop(masks_by_mod: dict, p_drop: float = 0.0, ensure_any_kept: bool = True):
    # Nothing to do if empty, no drop requested, or only one modality
    if not masks_by_mod or p_drop <= 0.0 or len(masks_by_mod) <= 1:
        return masks_by_mod

    first = next(iter(masks_by_mod.values()))
    B = first.shape[0]

    # Bernoulli drop per (batch, modality)
    keep_flags = {
        m: (torch.rand(B, device=first.device) > p_drop) for m in masks_by_mod.keys()
    }

    # Ensure we never drop the only present modality for a sample
    try:
        # presence map: True if any token kept for modality
        present = torch.stack([masks_by_mod[m].sum(dim=1) > 0 for m in masks_by_mod.keys()], dim=1)  # [B,M]
        present_counts = present.sum(dim=1)  # [B]
        for j, m in enumerate(masks_by_mod.keys()):
            # rows where this modality is the only present one
            only_this = (present_counts == 1) & present[:, j]
            if only_this.any():
                keep_flags[m][only_this] = True
    except Exception:
        pass

    if ensure_any_kept:
        # ensure at least one modality kept per sample
        # (if all are False for a sample, flip one to True)
        K = list(keep_flags.keys())
        stack = torch.stack([keep_flags[m] for m in K], dim=1)  # [B, M]
        all_dropped = ~stack.any(dim=1)                         # [B]
        if all_dropped.any():
            # arbitrarily keep the first modality for those samples
            keep_flags[K[0]][all_dropped] = True

    out = {}
    for m, mask in masks_by_mod.items():
        # If dropped, zero the whole attention mask; else keep as-is
        k = keep_flags[m].view(-1, 1).to(mask.dtype)            # [B,1]
        out[m] = mask * k                                       # broadcast over seq_len
    return out


def get_mask_rate_linear(epoch, total_epochs=20, mask_start=0.3, mask_end=0.05):
    decay_per_epoch = (mask_start - mask_end) / total_epochs
    mask_rate = mask_start - (decay_per_epoch * epoch)
    return max(mask_end, mask_rate)

def mask_tokens(
    inputs: Dict[str, Any],
    tokenizer,
    device: torch.device,
    mlm_probability: float = 0.15,
    mask_whole_alterations: bool = False,
    mask_whole_components: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Prepare masked tokens inputs/labels for masked language modeling.

    - Works with any tokenizer exposing pad_token_id, cls_token_id, mask_token_id.
    - If mask_whole_alterations=True, we sample positions from the 'gene' component
      and apply that same boolean mask to every discrete component (alignment by index).
    - If False, each discrete component is masked independently.
    - Non-discrete channels (e.g., floats like 'aa_vaf') are passed through unchanged.

    Returns:
        inputs_masked: Dict[str, Tensor] with masked IDs
        labels:        Dict[str, Tensor] with -100 on unmasked positions
    """
    inputs_masked: Dict[str, Any] = {}
    labels: Dict[str, torch.Tensor] = {}

    # Identify discrete components (LongTensor) and specials
    comp_names = [k for k, v in inputs.items() if isinstance(v, torch.Tensor) and v.dtype in (torch.long, torch.int64)]
    if len(comp_names) == 0:
        return inputs, { }

    pad_id = int(getattr(tokenizer, 'pad_token_id', 0))
    cls_id = int(getattr(tokenizer, 'cls_token_id', 1))
    mask_id = int(getattr(tokenizer, 'mask_token_id', pad_id))
    if mask_id == pad_id:
        raise ValueError(
            "Tokenizer is missing a distinct MASK token. Ensure SPECIALS['mask'] is included "
            "when constructing GeneTokenizer (e.g., via create_dna_tokenizer)."
        )

    # Base “not special” mask per component
    not_special: Dict[str, torch.Tensor] = {}
    for name in comp_names:
        x = inputs[name].to(device)
        not_special[name] = (~x.eq(pad_id)) & (~x.eq(cls_id))

    B, L = inputs[comp_names[0]].shape

    # Decide which positions to mask
    if mask_whole_alterations and ('gene' in comp_names):
        # One mask from 'gene' → broadcast to all comps
        xg = inputs['gene'].to(device)
        cand = (~xg.eq(pad_id)) & (~xg.eq(cls_id))
        # Bernoulli over positions (per token)
        pos_mask = (torch.rand(B, L, device=device) < float(mlm_probability)) & cand
        # Never mask CLS at position 0
        pos_mask[:, 0] = False
        for name in comp_names:
            inputs_masked[name] = inputs[name].clone().to(device)
            labels[name] = inputs[name].clone().to(device)
            labels[name][~pos_mask] = -100
            inputs_masked[name][pos_mask] = mask_id
    else:
        # Independent per component
        for name in comp_names:
            x = inputs[name].to(device)
            cand = not_special[name]
            if mask_whole_components:
                # Sample a single Bernoulli for the whole component per example, then apply to all valid positions
                take = (torch.rand(B, 1, device=device) < float(mlm_probability))
                pos_mask = take & cand
            else:
                pos_mask = (torch.rand(B, L, device=device) < float(mlm_probability)) & cand
            pos_mask[:, 0] = False  # never mask CLS
            inputs_masked[name] = x.clone()
            labels[name] = x.clone()
            labels[name][~pos_mask] = -100
            inputs_masked[name][pos_mask] = mask_id

    # Pass through non-discrete channels, but zero aa_vaf at masked positions
    # to prevent it from leaking the answer for aa_vaf_bin prediction.
    for name, val in inputs.items():
        if name not in comp_names:
            if name == 'aa_vaf' and isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                masked_val = val.clone().to(device)
                if pos_mask is not None:
                    masked_val[pos_mask] = 0.0
                inputs_masked[name] = masked_val
            else:
                inputs_masked[name] = val

    return inputs_masked, labels



def masking_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)  # [batch_size, seq_len]
    valid_mask = labels != -100  # [batch_size, seq_len]
    correct = (preds == labels) & valid_mask  # [batch_size, seq_len]
    correct_sum = correct.sum().item()
    masked_sum = valid_mask.sum().item()
    if masked_sum == 0:
        return 0.0
    else:
        return (correct_sum / masked_sum)

def finite_reduce_loss(losses):
    loss = sum(losses)
    finite_mask = torch.isfinite(loss)
    finite_losses = loss[finite_mask]
    return finite_losses.mean()


def permute_batch(inputs, attention_mask, permute_keys, probability=0.15):
    """
    Randomly permutes tokens (excluding the first token) up to the first padding token 
    for each sample in the batch, consistently across specified keys, based on a shuffle probability.
    
    Args:
        inputs (dict): Dictionary of input tensors to permute.
        attention_mask (torch.Tensor): Attention mask tensor (batch_size, seq_length).
        permute_keys (list): Keys in inputs to apply permutation.
        probability (float): Probability of shuffling each sample.
        
    Returns:
        dict: Permuted inputs.
    """
    batch_size, seq_length = attention_mask.size()
    
    permuted_inputs = {}
    for key in inputs:
        if key in permute_keys:
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                permuted_inputs[key] = value.clone()
            elif isinstance(value, np.ndarray):
                permuted_inputs[key] = cp.deepcopy(value)
        else:
            permuted_inputs[key] = inputs[key]

    rand_vals = torch.rand(batch_size, device=attention_mask.device)
    permute_mask = rand_vals < probability 
    
    permute_indices = torch.nonzero(permute_mask, as_tuple=True)[0]
    
    if permute_indices.numel() == 0:
        return permuted_inputs
    
    for i in permute_indices:
        valid_mask = attention_mask[i].bool()
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        
        if valid_indices.numel() > 1:
            token_indices = valid_indices[1:]  # Exclude CLS token
            if token_indices.numel() == 0:
                continue  
            
            shuffled_indices = token_indices[torch.randperm(token_indices.size(0))]
            
            for key in permute_keys:
                if isinstance(permuted_inputs[key], torch.Tensor):
                    if permuted_inputs[key].dim() >= 2:
                        permuted_inputs[key][i, token_indices] = inputs[key][i, shuffled_indices]
                elif isinstance(permuted_inputs[key], np.ndarray):
                    if permuted_inputs[key].ndim >= 2:
                        permuted_inputs[key][i.cpu().numpy(), token_indices.cpu().numpy()] = inputs[key][i.cpu().numpy(), shuffled_indices.cpu().numpy()]
                        
    return permuted_inputs

