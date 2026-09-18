import os
import copy as cp
import math
import json
import hashlib
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from oncoformer.checkpoints import load_stream_backbone
from omegaconf import OmegaConf
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from oncoformer.training import (
    get_mask_rate_linear,
    mask_tokens,
    masking_accuracy,
    finite_reduce_loss,
    _build_keep_mask,
    _maybe_moddrop,
    permute_batch, 
)
from oncoformer.tokenizer import GeneTokenizer, SPECIALS
from oncoformer.gremln_kernels import _chebyshev_diffusion

_STR2DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

################################################################################
# Base Encoders

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class IntegerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, max_value: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_value = max_value
        self.register_buffer('div_term', torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        x = x.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        position = x / self.max_value
        pe = torch.zeros(x.size(0), x.size(1), self.embedding_dim).to(x.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe

class FourierFractionEncoder(nn.Module):
    def __init__(self, embedding_dim=128, fourier_dim=32, f_min=1.0, f_max=64.0,
                 add_logit=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("freqs", torch.logspace(math.log10(f_min), math.log10(f_max), fourier_dim))
        self.add_logit = add_logit
        in_dim = 2*fourier_dim + 1 + (1 if add_logit else 0)  # sin/cos + raw + (logit)
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, embedding_dim))

    def forward(self, x):  # x in [0,1]
        x = x.clamp(0, 1)
        phases = 2*math.pi * x.unsqueeze(-1) * self.freqs                    # [B,L,K]
        fe = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)       # [B,L,2K]
        parts = [fe, x.unsqueeze(-1)]
        if self.add_logit:
            eps = 1e-5
            parts.append(torch.log((x+eps)/(1-x+eps)).unsqueeze(-1))
        return self.proj(torch.cat(parts, dim=-1)) 

class FourierFiLMGateEncoder(nn.Module):
    """
    Expression co-encoder that augments the target embedding with Fourier features,
    applies FiLM conditioning, and gates the result based on expression fractions.
    """
    def __init__(
        self,
        embedding_dim: int = 512,
        log_range: float = math.log(8.0),
        *,
        add_fourier: bool = True,
        fourier_dim: int = 32,
        fourier_add_logit: bool = True,
        film: bool = True,
        film_hidden_dim: int = 128,
        film_use_logit: bool = True,
        **unused: Any,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.log_range = float(log_range)
        self.add_fourier = bool(add_fourier)
        self.film_enabled = bool(film)
        self.film_use_logit = bool(film_use_logit)

        if self.add_fourier:
            self.fourier_encoder = FourierFractionEncoder(
                embedding_dim=self.embedding_dim,
                fourier_dim=fourier_dim,
                f_min=1.0,
                f_max=64.0,
                add_logit=fourier_add_logit,
            )
        else:
            self.register_module("fourier_encoder", None)

        if self.film_enabled:
            film_input_dim = 1 + (1 if film_use_logit else 0)
            self.film_mlp = nn.Sequential(
                nn.Linear(film_input_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, 2 * self.embedding_dim),
            )
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
        else:
            self.register_module("film_mlp", None)

    @staticmethod
    def _logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return torch.log((x + eps) / (1.0 - x + eps))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.clamp(0, 1)
        outputs: Dict[str, torch.Tensor] = {}

        if self.add_fourier and self.fourier_encoder is not None:
            outputs["add"] = self.fourier_encoder(x)

        if self.film_enabled and self.film_mlp is not None:
            features = [x.unsqueeze(-1)]
            if self.film_use_logit:
                features.append(self._logit(x).unsqueeze(-1))
            film_input = torch.cat(features, dim=-1)
            film_out = self.film_mlp(film_input)
            scale, shift = film_out.chunk(2, dim=-1)
            outputs["film_scale"] = scale
            outputs["film_shift"] = shift

        gate = torch.exp(self.log_range * (2.0 * x - 1.0)).unsqueeze(-1)
        outputs["gate"] = gate
        return outputs

class LayerNormEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x
    
################################################################################
# Posttraining Prediction heads (pooled embedding -> prediction only)

class SimpleLinear(L.LightningModule):
    """
    Drop-in head that operates on pooled embeddings only: (B, D) -> (B, output_dim)
    - Optional hidden layer
    - Dropout before each Linear
    - Optional LayerNorm/BatchNorm on the input features
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        loss_fn = nn.MSELoss(),
        *,
        dropout: float = 0.3,
        hidden_dim: Optional[int] = None,
        norm: Optional[Literal["layernorm","batchnorm"]] = None,
        activation: Literal["gelu","relu","silu"] = "gelu",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.loss_fn = loss_fn

        # optional input normalization
        if norm == "layernorm":
            self.input_norm = nn.LayerNorm(input_dim)
        elif norm == "batchnorm":
            self.input_norm = nn.BatchNorm1d(input_dim)
        else:
            self.input_norm = None

        act = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[activation]

        if hidden_dim is None:
            # single linear with a dropout on inputs
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
            )
        else:
            # tiny MLP with dropout
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already pooled: shape (B, D)
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.net(x)

    def _common_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        # batch = (pooled_embeddings, targets)
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss


class Regression(SimpleLinear):
    """
    Multi-target regression from pooled embedding.
    Default loss = MSE; switch to Huber via use_huber=True for outlier robustness.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.15,
        hidden_dim: Optional[int] = None,
        norm: Optional[Literal["layernorm","batchnorm"]] = None,
        use_huber: bool = False,
        huber_delta: float = 1.0,
    ):
        loss = nn.SmoothL1Loss(beta=huber_delta) if use_huber else nn.MSELoss()
        super().__init__(input_dim, output_dim, loss_fn=loss,
                         dropout=dropout, hidden_dim=hidden_dim, norm=norm)


class Classification(SimpleLinear):
    """
    Single-label multiclass classification from pooled embedding.
    - Respects ignore_index=-100 (OncoformerPost can set ignored labels to -100)
    - Label smoothing default 0.05 (tunable)
    - Optional class_weights (1D tensor) for imbalance
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.3,
        hidden_dim: Optional[int] = None,
        norm: Optional[Literal["layernorm","batchnorm"]] = None,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
    ):
        ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            weight=class_weights,
        )
        super().__init__(input_dim, output_dim, loss_fn=ce,
                         dropout=dropout, hidden_dim=hidden_dim, norm=norm)

    
################################################################################
# Posttraining Baseline heads
    
class PassThroughEncoder(L.LightningModule):
    def __init__(self, raw_value_key: str, n_components: int):
        super().__init__()
        self.config = {
            'raw_value_key': raw_value_key,
            'n_components': n_components,
        }
        self.embed_dim = n_components
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.config['n_components']]
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        results = self(batch[self.config['raw_value_key']]['values'])
        return {'sample_embeddings': results}
    
class PCAEncoder(L.LightningModule):
    def __init__(self, dataloader: DataLoader, raw_value_key: str, n_components: int, seed: int = 0):
        super().__init__()
        self.config = {
            'raw_value_key': raw_value_key,
            'n_components': n_components,
            'seed': seed,
        }
        self.embed_dim = n_components
        
        raw_values = torch.concat([batch[self.config['raw_value_key']]['values'] for batch in dataloader], dim=0)
        raw_values = raw_values.detach().cpu().numpy()
        
        np.random.seed(self.config['seed'])
        scaler = StandardScaler()
        raw_values_scaled = scaler.fit_transform(raw_values)
        pca = PCA(n_components=self.config['n_components'])
        raw_values_pca = pca.fit_transform(raw_values_scaled)

        scaler_scale = torch.from_numpy(scaler.scale_).float()
        scaler_mean = torch.from_numpy(scaler.mean_).float()
        pca_components = torch.from_numpy(pca.components_).float()
        pca_mean = torch.from_numpy(pca.mean_).float()

        self.register_buffer('scaler_scale', scaler_scale)
        self.register_buffer('scaler_mean', scaler_mean)
        self.register_buffer('pca_components', pca_components)
        self.register_buffer('pca_mean', pca_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = (x - self.scaler_mean) / self.scaler_scale        
        return torch.matmul(x_scaled - self.pca_mean, self.pca_components.T)
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        results = self(batch[self.config['raw_value_key']]['values'])
        return {'sample_embeddings': results}

################################################################################

class PrecomputedEmbeddingsEncoder(L.LightningModule):
    """
    Loads a precomputed embedding table:
      - weight_path: Float/half Tensor [V, D] (torch.save)
      - vocab_path : JSON with {"token_to_idx", "specials", "embedding_dim", ...}

    trainable=False → F.embedding on a buffer (fast, frozen)
    trainable=True  → nn.Embedding initialized from weight

    Forward: LongTensor [B, L] → FloatTensor [B, L, D]
    """
    def __init__(
        self,
        weight_path: str,
        vocab_path: str,
        *,
        trainable: bool = False,
        normalize: bool = True,
        dtype: str = "fp16",      # "fp16" | "bf16" | "fp32"
        **kwargs,
    ):
        super().__init__()
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        # Always load on CPU to avoid GPU spikes; cast afterwards
        W = torch.load(weight_path, map_location='cpu')
        if isinstance(W, dict) and "weight" in W:
            W = W["weight"]
        target_dtype = _STR2DTYPE.get(dtype, torch.float16)
        if W.dtype != target_dtype:
            W = W.to(target_dtype)   # cast on load

        V, D = W.shape
        self.embedding_dim = D
        self.vocab_size = V
        self.normalize = bool(normalize)

        specials = vocab.get('specials', {})
        self.pad_idx = vocab['token_to_idx'].get(specials.get('pad', '<pad>'), 0)

        if trainable:
            # Keep the dtype of W when creating the trainable table
            self.weight = None
            W_fp32 = W.float()
            self.embedding = nn.Embedding.from_pretrained(
                W_fp32, freeze=False, padding_idx=self.pad_idx
            )
        else:
            self.embedding = None
            self.register_buffer('weight', W)

        # Keep LN params in float32; cast as needed at runtime (see _maybe_norm)
        self.enc_norm = nn.LayerNorm(D)

    def _maybe_norm(self, out: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return out
        # On CPU, half/bfloat16 LayerNorm can be slow/unsupported; do it in fp32 then cast back
        if out.device.type == "cpu" and out.dtype in (torch.float16, torch.bfloat16):
            return self.enc_norm(out.float()).to(out.dtype)
        return self.enc_norm(out)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dtype != torch.long:
            ids = ids.long()
        if self.embedding is not None:
            out = self.embedding(ids)
        else:
            out = F.embedding(ids, self.weight)
        return self._maybe_norm(out)



class PrecomputedEmbeddingsAtomicEncoder(PrecomputedEmbeddingsEncoder):
    """
    Same as PrecomputedEmbeddingsEncoder, but a subset of rows are 'atomic'
    (e.g., amplification/deletion). Those rows get a small trainable overlay.

    Vocab JSON (one or both):
      - "atomic_rows":   [int...]
      - "atomic_tokens": [str...]  (resolved via token_to_idx)

    Params override (optional via kwargs):
      - atomic_tokens: [str...]
      - trainable_base: bool
      - trainable_atomic: bool
      - dtype: "fp16"|"bf16"|"fp32"
    """
    def __init__(
        self,
        weight_path: str,
        vocab_path: str,
        *,
        trainable_base: bool = False,
        trainable_atomic: bool = True,
        dtype: Union[str, int, torch.dtype] = "fp16",   # keep your dtype support
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(weight_path, vocab_path, trainable=trainable_base, normalize=normalize, dtype=dtype)

        # Reuse parent's storage as the "base"
        if self.embedding is not None:  # trainable base
            self.base = self.embedding
            self.embedding = None
            self.register_buffer('weight', None)
        else:                            # frozen buffer base
            self.base = None
            # self.weight already registered by parent

        # read vocab once (small)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        t2i = vocab.get("token_to_idx", {})
        specials = vocab.get("specials", SPECIALS)
        pad_idx = int(t2i.get(specials.get("pad", SPECIALS["pad"]), 0))

        V, D = self.vocab_size, self.embedding_dim

        # discover atomic rows
        atomic_rows = vocab.get("atomic_rows", None)
        if atomic_rows is None and "atomic_tokens" in vocab:
            atomic_rows = [t2i[tok] for tok in vocab["atomic_tokens"] if tok in t2i]
        atomic_rows = sorted({int(r) for r in (atomic_rows or []) if 0 <= int(r) < V})

        row_to_atomic = torch.full((V,), -1, dtype=torch.long)
        for j, r in enumerate(atomic_rows):
            row_to_atomic[r] = j
        self.register_buffer("row_to_atomic", row_to_atomic)
        self.num_atomic = len(atomic_rows)


        # Determine the base dtype (fp16/bf16/fp32)
        base_dtype = self.base.weight.dtype if self.base is not None else self.weight.dtype

        # small trainable overlay just for atomic rows
        self.atomic = None
        if self.num_atomic > 0:
            self.atomic = nn.Embedding(self.num_atomic, D)
            # ensure dtype matches the base table
            self.atomic = self.atomic.to(dtype=base_dtype)
            nn.init.normal_(self.atomic.weight, mean=0.0, std=0.02)
            self.atomic.weight.requires_grad_(bool(trainable_atomic))


    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dtype != torch.long:
            ids = ids.long()

        # base lookup (unnormalized)
        if self.base is not None:
            out = self.base(ids)
        else:
            out = F.embedding(ids, self.weight)

        # atomic overlay
        if self.atomic is not None and self.num_atomic > 0:
            atomic_idx = self.row_to_atomic[ids]
            mask = atomic_idx >= 0
            if mask.any():
                rep = self.atomic(atomic_idx[mask])
                if rep.dtype != out.dtype:
                    rep = rep.to(out.dtype)
                out = out.clone()
                out.view(-1, out.size(-1))[mask.view(-1)] = rep


        return self.enc_norm(out) if self.normalize else out



################################################################################
# Omics Encoders

class ModalityEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        pad_token_id: int = 0,
        na_token_id: int = None,
        component_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.na_token_id = na_token_id
        
        self.encoders = nn.ModuleDict()
        self.combination_strategies = {}
        self.embedding_ranges = {}
        self.loss_param = {}
        self.gate_configs: Dict[str, Dict[str, Any]] = {}
                
        if component_configs is not None:
            for comp_name, config in component_configs.items():
                # Always record loss weights, even if the component is not encoded.
                self.loss_param[comp_name] = config.get('loss_param', 1.0)

                if not config.get('encode', True):
                    continue
                encoder_type = config['encoder_type']
                params = config['params']
                if 'embedding_dim' not in params:
                    params['embedding_dim'] = embedding_dim
                encoder = encoder_type(**params)
                self.encoders[comp_name] = encoder
                self.combination_strategies[comp_name] = config.get('combination', 'sum')
                if 'embedding_range' in config:
                    self.embedding_ranges[comp_name] = config['embedding_range']
                if self.combination_strategies[comp_name] == 'gate':
                    targets = config.get('gate_targets')
                    if not targets:
                        targets = ['__combined__']
                    self.gate_configs[comp_name] = {
                        'targets': list(targets),
                        'mode': str(config.get('gate_mode', 'mul')).lower(),
                    }
        
        # Partial sum?
        self.partial_sum = any(
            strategy == 'partial_sum' for strategy in self.combination_strategies.values()
        )
        if self.partial_sum:
            # Initialize zero vector to accumulate embeddings
            self.register_buffer('zero_embedding', torch.zeros(1, 1, embedding_dim))
            
        # Detect if 'concat' is used and create projection layer
        self.concat_encodings = any(
            strategy == 'concat' for strategy in self.combination_strategies.values()
        )
        if self.concat_encodings:
            total_embedding_dim = 0
            for comp_name, strategy in self.combination_strategies.items():
                if strategy == 'concat':
                    encoder = self.encoders[comp_name]
                    total_embedding_dim += encoder.embedding_dim
            # Projection layer to project back to embedding_dim
            self.projection_layer = nn.Linear(total_embedding_dim, embedding_dim)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        
        first_key = next(iter(inputs))
        batch_size, seq_length = inputs[first_key].shape
        device = inputs[first_key].device

        concat_embeds_list: List[torch.Tensor] = []
        sum_embeds = torch.zeros(batch_size, seq_length, self.embedding_dim, device=device)
        encoded_components: Dict[str, torch.Tensor] = {}
        gate_values: Dict[str, Dict[str, torch.Tensor]] = {}
        combined_payloads: List[Dict[str, torch.Tensor]] = []

        for comp_name, input_ids in inputs.items():
            if comp_name not in self.encoders:
                continue

            encoder = self.encoders[comp_name]
            embeds = encoder(input_ids)

            if (self.na_token_id is not None) and isinstance(input_ids, torch.Tensor):
                na_mask = input_ids.eq(self.na_token_id).unsqueeze(-1)  # (batch_size, seq_length, 1)
                embeds = embeds.masked_fill(na_mask, 0.0)

            combination = self.combination_strategies.get(comp_name, 'sum')
            if combination == 'gate':
                payload = embeds if isinstance(embeds, dict) else {'gate': embeds}
                gate_values[comp_name] = payload
            else:
                encoded_components[comp_name] = embeds

        for gate_name, gate_tensor in gate_values.items():
            cfg = self.gate_configs.get(gate_name, {})
            targets = cfg.get('targets') or ['__combined__']
            mode = str(cfg.get('mode', 'mul')).lower()

            for target in targets:
                if target == '__combined__':
                    combined_payloads.append({'mode': mode, **gate_tensor})
                    continue

                if target not in encoded_components:
                    continue

                payload = gate_tensor
                target_embed = encoded_components[target]

                add_tensor = payload.get('add')
                if add_tensor is not None:
                    target_embed = target_embed + add_tensor.to(target_embed.dtype)

                scale_tensor = payload.get('film_scale')
                if scale_tensor is not None:
                    scale_tensor = scale_tensor.to(target_embed.dtype)
                    target_embed = target_embed * (1.0 + scale_tensor)

                shift_tensor = payload.get('film_shift')
                if shift_tensor is not None:
                    target_embed = target_embed + shift_tensor.to(target_embed.dtype)

                gate_component = payload.get('gate')
                if gate_component is not None:
                    gate_comp = gate_component
                    if gate_comp.size(-1) == 1 and target_embed.size(-1) != 1:
                        gate_comp = gate_comp.expand_as(target_embed)
                    elif gate_comp.size(-1) not in (1, target_embed.size(-1)):
                        raise ValueError(
                            f"Gate '{gate_name}' width {gate_comp.size(-1)} incompatible with target "
                            f"'{target}' width {target_embed.size(-1)}"
                        )
                    gate_comp = gate_comp.to(target_embed.dtype)
                    if mode == 'mul':
                        target_embed = target_embed * gate_comp
                    else:
                        raise ValueError(f"Unsupported gate mode '{mode}' for gate '{gate_name}'")

                encoded_components[target] = target_embed

        for comp_name, embeds in inputs.items():
            if comp_name not in encoded_components:
                continue

            embeds = encoded_components[comp_name]
            combination = self.combination_strategies.get(comp_name, 'sum')
            if combination == 'sum':
                sum_embeds += embeds
            elif combination == 'concat':
                if embeds.dtype != sum_embeds.dtype:
                    embeds = embeds.to(sum_embeds.dtype)
                concat_embeds_list.append(embeds)
            elif combination == 'partial_sum':
                embedding_range = self.embedding_ranges[comp_name]
                start_idx, end_idx = embedding_range
                sum_embeds[:, :, start_idx:end_idx] += embeds
            else:
                raise ValueError(f"Unknown combination strategy: {combination}")
        
        if self.concat_encodings:
            combined_embeds = torch.cat(concat_embeds_list, dim=-1) if concat_embeds_list else torch.zeros_like(sum_embeds)
            combined_embeds = self.projection_layer(combined_embeds)

            if (sum_embeds != 0).any():
                combined_embeds += sum_embeds
        else:
            combined_embeds = sum_embeds

        for payload in combined_payloads:
            add_tensor = payload.get('add')
            if add_tensor is not None:
                combined_embeds = combined_embeds + add_tensor.to(combined_embeds.dtype)

        for payload in combined_payloads:
            scale_tensor = payload.get('film_scale')
            if scale_tensor is not None:
                combined_embeds = combined_embeds * (1.0 + scale_tensor.to(combined_embeds.dtype))
            shift_tensor = payload.get('film_shift')
            if shift_tensor is not None:
                combined_embeds = combined_embeds + shift_tensor.to(combined_embeds.dtype)

        for payload in combined_payloads:
            gate_tensor = payload.get('gate')
            if gate_tensor is None:
                continue
            gate = gate_tensor
            if gate.size(-1) == 1:
                gate = gate.expand_as(combined_embeds)
            elif gate.size(-1) != combined_embeds.size(-1):
                raise ValueError(
                    f"Gate targeting '__combined__' expects width {combined_embeds.size(-1)}, "
                    f"got {gate.size(-1)}"
                )
            gate = gate.to(combined_embeds.dtype)
            if payload.get('mode', 'mul') == 'mul':
                combined_embeds = combined_embeds * gate
            else:
                raise ValueError(f"Unsupported gate mode '{payload.get('mode')}' for final combination")
        
        return combined_embeds

class ModalityMLM(nn.Module):
    def __init__(
        self,
        modality_encoder: nn.Module,
        component_configs: Dict[str, Dict[str, Any]],
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_length: int = 128, 
        pad_idx: int = 0,
        save_self_attn: bool = False,
    ):
        super(ModalityMLM, self).__init__()
        
        self.modality_encoder = modality_encoder 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
            
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.attn = [SaveAttention() for _ in range(self.transformer_encoder.num_layers)]
        self._attn_handles: List[Optional[Any]] = []
        self._configure_attention_hooks(save_self_attn)

        self.max_length = max_length
        
        self.pad_idx = pad_idx
        
        # MLM Heads for each component with support for alias routing
        self.mlm_heads = nn.ModuleDict()
        self.predict_alias_map: Dict[str, List[str]] = {}

        target_heads: Dict[str, int] = {}

        for comp_name, comp_config in component_configs.items():
            predict_setting = comp_config.get('predict', False)
            if predict_setting is False or predict_setting is None:
                continue

            if predict_setting is True:
                targets = [comp_name]
            elif isinstance(predict_setting, str):
                targets = [predict_setting]
            elif isinstance(predict_setting, Iterable):
                targets = [str(t) for t in predict_setting]
            else:
                raise TypeError(f"Unsupported predict setting for component '{comp_name}': {type(predict_setting)}")

            targets_unique = list(dict.fromkeys(targets))
            if predict_setting is not True:
                self.predict_alias_map[comp_name] = targets_unique

            for tgt in targets_unique:
                tgt_cfg = component_configs.get(tgt)
                if tgt_cfg is None:
                    raise KeyError(f"Predict target '{tgt}' referenced by component '{comp_name}' is not defined.")
                vocab_size = int(tgt_cfg['num_embeddings'])
                if tgt in target_heads and target_heads[tgt] != vocab_size:
                    raise ValueError(
                        f"Conflicting vocab sizes for predict target '{tgt}': "
                        f"{target_heads[tgt]} vs {vocab_size}"
                    )
                target_heads[tgt] = vocab_size

        for head_name, vocab_size in target_heads.items():
            self.mlm_heads[head_name] = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for mlm_head in self.mlm_heads.values():
            nn.init.xavier_uniform_(mlm_head.weight)
            if mlm_head.bias is not None:
                nn.init.constant_(mlm_head.bias, 0)
            
    def _configure_attention_hooks(self, enabled: bool) -> None:
        for handle in getattr(self, "_attn_handles", []):
            if handle is not None:
                handle.remove()
        self._attn_handles = []
        self._save_self_attn = bool(enabled)
        for idx, layer in enumerate(self.transformer_encoder.layers):
            patch_attention(layer.self_attn, need_weights=self._save_self_attn)
            if self._save_self_attn:
                self.attn[idx].clear()
                handle = layer.self_attn.register_forward_hook(self.attn[idx])
            else:
                self.attn[idx].clear()
                handle = None
            self._attn_handles.append(handle)

    def enable_attention_capture(self) -> None:
        if self._save_self_attn:
            self.clear_attention_buffers()
            return
        self._configure_attention_hooks(True)

    def disable_attention_capture(self) -> None:
        if not self._save_self_attn:
            return
        self._configure_attention_hooks(False)

    @property
    def attention_capture_enabled(self) -> bool:
        return getattr(self, "_save_self_attn", False)

    def clear_attention_buffers(self) -> None:
        for hook in self.attn:
            hook.clear()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary containing input IDs for each component.
                - 'gene': Tensor of shape (batch_size, seq_len)
                - Other components (e.g., 'modality', 'value', etc.): Tensors of shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_len) where 1 indicates valid tokens and 0 indicates padding.
        Returns:
            Dict[str, torch.Tensor]: Dictionary of logits for each component.
                - Each tensor has shape (batch_size, seq_len, vocab_size),
            embeddings:  Shape: (batch_size, seq_len, embed_dim)
        """

        embeddings = self.modality_encoder(inputs) 
        
        encoded_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=~attention_mask.bool()
        )  # Shape: (batch_size, seq_len, embed_dim)
        
        logits = {}
        for comp_name, mlm_head in self.mlm_heads.items():
            logits[comp_name] = mlm_head(encoded_output)  # Shape: (batch_size, seq_len, vocab_size)
        
        return logits, encoded_output

    @property
    def layers(self):
        # alias for convenience
        return self.transformer_encoder.layers

    def initial_embed(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [B, L, D]
        return self.modality_encoder(inputs)

    def encode_tokens_step(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        graph_data: Optional[Dict[str, Any]] = None,
        graph_cfg: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        norm_first = getattr(layer, "norm_first", False)
        key_padding_mask = ~attention_mask.bool()

        if norm_first:
            attn_input = layer.norm1(hidden)
        else:
            attn_input = hidden

        query_input = self._apply_graph_diffusion(
            attn_input,
            graph_data,
            graph_cfg,
            num_heads=layer.self_attn.num_heads,
        )

        attn_output, _ = layer.self_attn(
            query_input,
            attn_input,
            attn_input,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=layer.self_attn.batch_first,
        )
        attn_output = layer.dropout1(attn_output)

        if norm_first:
            x = hidden + attn_output
            ff_input = layer.norm2(x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            ff_output = layer.dropout2(ff_output)
            x = x + ff_output
        else:
            x = layer.norm1(hidden + attn_output)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            ff_output = layer.dropout2(ff_output)
            x = layer.norm2(x + ff_output)

        return x

    def _apply_graph_diffusion(
        self,
        src: torch.Tensor,
        graph_data: Optional[Dict[str, Any]],
        graph_cfg: Optional[Dict[str, Any]],
        num_heads: int = 1,
    ) -> torch.Tensor:
        if (
            graph_cfg is None
            or not graph_cfg.get('enabled', False)
            or graph_data is None
        ):
            return src

        extra_nodes = int(graph_data.get('extra_nodes', 0) or 0)
        core_nodes = graph_data.get('core_nodes', None)

        edge_index_list = graph_data.get('edge_index_list', None)
        num_nodes_list = graph_data.get('num_nodes_list', None)
        if not edge_index_list or not num_nodes_list:
            return src
        if len(edge_index_list) != src.size(0) or len(num_nodes_list) != src.size(0):
            return src

        device = src.device
        edge_index_list = [ei.to(device=device) for ei in edge_index_list]
        edge_weight_list = graph_data.get('edge_weight_list', None)
        if edge_weight_list is not None:
            edge_weight_list = [
                (ew.to(device=device) if ew is not None else None)
                for ew in edge_weight_list
            ]

        seq_len = src.size(1)
        preserve_cls = bool(graph_cfg.get('preserve_cls', True))
        cheby_K = int(graph_cfg.get('cheby_K', 6))
        beta = float(graph_cfg.get('beta', 0.5))

        cls_offset = 0
        expected_without_cls = seq_len
        if preserve_cls and seq_len > 1:
            cls_offset = 1
            expected_without_cls = seq_len - 1

        expected_total = expected_without_cls + extra_nodes
        if not all((n == expected_total) for n in num_nodes_list):
            return src

        if cls_offset:
            slice_tokens = src[:, cls_offset:, :]
            if slice_tokens.size(1) == 0:
                return src
        else:
            slice_tokens = src

        if extra_nodes > 0:
            zeros = torch.zeros(
                slice_tokens.size(0),
                extra_nodes,
                slice_tokens.size(2),
                dtype=slice_tokens.dtype,
                device=slice_tokens.device,
            )
            slice_tokens = torch.cat([slice_tokens, zeros], dim=1)

        head_dim = slice_tokens.size(-1) // num_heads
        if head_dim * num_heads != slice_tokens.size(-1):
            return src

        reshaped = slice_tokens.contiguous().view(
            slice_tokens.size(0),
            slice_tokens.size(1),
            num_heads,
            head_dim,
        )
        diffused = _chebyshev_diffusion(
            edge_index_list=edge_index_list,
            num_nodes_list=num_nodes_list,
            E=reshaped.to(dtype=torch.float32),
            k=cheby_K,
            beta=beta,
            edge_weight_list=edge_weight_list,
        ).to(dtype=src.dtype)
        diffused = diffused.view(slice_tokens.size(0), slice_tokens.size(1), -1)

        if extra_nodes > 0:
            diffused = diffused[:, :expected_without_cls, :]

        if cls_offset:
            return torch.cat([src[:, :cls_offset, :], diffused], dim=1)
        return diffused

    def apply_mlm_heads(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        # apply all MLM heads on hidden (same as forward's head stage)
        return {comp: head(hidden) for comp, head in self.mlm_heads.items()}

################################################################################
# Attention and Fusion models

def patch_attention(m, need_weights: bool = True):
    forward_orig = getattr(m, "_oncoformer_forward_orig", m.forward)
    if getattr(m, "_oncoformer_need_weights", None) == need_weights:
        return
    capture_need_weights = need_weights

    def wrap(query, key, value, key_padding_mask=None,
             need_weights=True, average_attn_weights=False, **kwargs):
        # Guard: if a row is fully masked, unmask the first token and zero it out
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)  # [B]
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key = key.clone()
                value = value.clone()
                key_padding_mask[all_masked, 0] = False
                key[all_masked, 0, :].zero_()
                value[all_masked, 0, :].zero_()

        return forward_orig(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=capture_need_weights,
            average_attn_weights=False,
            **kwargs
        )

    m.forward = wrap
    m._oncoformer_forward_orig = forward_orig
    m._oncoformer_need_weights = need_weights


class SaveAttention:
    def __init__(self):
        self.data = None

    def __call__(self, module, module_in, module_out):
        self.data = module_out[1]

    def clear(self):
        self.data = None

class DeepCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, pattern='cls2all', ff_ratio=4.0, gate_init=-2.0):
        super().__init__()
        self.pattern = pattern
        self.topk = None
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(ff_ratio*d_model)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(ff_ratio*d_model), d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full([], gate_init))
        self.last_attn = None  # optional

    def _select_q(self, Hq):
        if self.pattern in ('cls2cls', 'cls2all'):
            return Hq[:, :1, :], 'cls'  # [B,1,D]
        if self.pattern == 'all2all':
            return Hq, 'all'            # [B,L,D]
        raise ValueError(self.pattern)

    def _select_kv(self, Hkv, mask_kv=None, salience=None):
        if self.pattern == 'cls2cls':
            return Hkv[:, :1, :], None  # [B,1,D]
        if self.topk is None:
            kpm = None if mask_kv is None else ~mask_kv.bool()  # True means pad
            KV = Hkv
            if kpm is not None:
                # If a whole row is padded, unmask the first token and zero its value.
                all_pad = kpm.all(dim=1)  # [B]
                if all_pad.any():
                    KV = KV.clone()
                    KV[all_pad, :1, :].zero_()
                    kpm = kpm.clone()
                    kpm[all_pad, :1] = False
            return KV, kpm
        # Top-K path unchanged
        B, L, D = Hkv.shape
        if salience is None:
            salience = Hkv.norm(dim=-1)  # [B, L]
        if mask_kv is not None:
            salience = salience.masked_fill(~mask_kv.bool(), float('-inf'))
        k = min(self.topk, L)
        idx = torch.topk(salience, k=k, dim=1, largest=True).indices  # [B, k]
        ar = torch.arange(B, device=Hkv.device).unsqueeze(-1)
        Hsel = Hkv[ar, idx]  # [B, k, D]
        kpm = Hsel.new_zeros((B, k), dtype=torch.bool)
        return Hsel, kpm


    def forward(self, Hq, Hkv, mask_q=None, mask_kv=None, salience_kv=None):
        Q, _ = self._select_q(Hq)
        KV, key_padding_mask = self._select_kv(Hkv, mask_kv, salience=salience_kv)

        # if a whole row is masked, unmask position 0 and zero its value ---
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)  # [B]
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                KV = KV.clone()
                key_padding_mask[all_masked, 0] = False
                KV[all_masked, 0, :].zero_()

        out, w = self.attn(
            Q, KV, KV,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False
        )
        self.last_attn = w

        delta = self.ff(self.drop(self.norm_q(out)))
        Hq = Hq.clone()
        g = torch.sigmoid(self.gate)
        if self.pattern in ('cls2cls', 'cls2all'):
            Hq[:, :1, :] = Hq[:, :1, :] + g * delta
        else:
            Hq = Hq + g * delta
        return Hq



class DeepCrossStitcher(nn.Module):
    def __init__(self, modalities, d_model, nhead, dropout=0.1,
                 pattern='cls2all', ff_ratio=4.0, gate_init=-2.0):
        super().__init__()
        self.modalities = list(modalities)
        self.blocks = nn.ModuleDict()
        for mi in self.modalities:
            for mj in self.modalities:
                if mi == mj: continue
                self.blocks[f"{mi}<-{mj}"] = DeepCrossAttention(
                    d_model, nhead, dropout, pattern, ff_ratio, gate_init
                )
        self.topk = None  # may set globally or per-pair

    def forward(self, H: dict, M: dict, salience: dict = None):
        for tgt in self.modalities:
            for src in self.modalities:
                if src == tgt: continue
                key = f"{tgt}<-{src}"
                block = self.blocks[key]
                if isinstance(self.topk, dict):
                    block.topk = self.topk.get(key, None)
                else:
                    block.topk = self.topk
                H[tgt] = block(H[tgt], H[src], mask_q=M.get(tgt), mask_kv=M.get(src),
                               salience_kv=None if salience is None else salience.get(src))
        return H


class CLIPMoCoHead(nn.Module):
    """Projection head for CLIP-MoCo cross-modal contrastive learning."""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)  # [B, out_dim] L2-normalized


################################################################################
# Oncoformer Models

class LitOncoformerBase(L.LightningModule):
    def __init__(self, config: Dict[str, Any], checkpoint_dir: str):
        super().__init__()
        self.config = cp.deepcopy(config)
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        
    def get_config_hash(self) -> str:
        serialized_config =  OmegaConf.to_yaml(self.config, sort_keys=True)
        return hashlib.md5(serialized_config.encode()).hexdigest()[:8]
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizer_type = self.config['training'].get('optimizer', 'AdamW')
        learning_rate = self.config['training'].get('learning_rate', 1e-4)
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        ## This section needs to be updated
        ## None for auto-detect, 10% for warm-up
        training_cfg = self.config.setdefault('training', {})
        training_len = int(training_cfg.get('training_len', 1))
        if training_len <= 0:
            training_len = 1
        training_cfg['training_len'] = training_len

        num_epochs = int(training_cfg.get('num_epochs', 20))
        total_steps = max(training_len * num_epochs, 1)

        warmup_steps = training_cfg.get('scheduler_warmup_steps')
        if warmup_steps is None:
            warmup_fraction = float(training_cfg.get('scheduler_warmup_fraction', 0.1))
            warmup_steps = int(total_steps * warmup_fraction)
            training_cfg['scheduler_warmup_steps'] = warmup_steps
        warmup_steps = int(warmup_steps)

        scheduler_type = training_cfg.get('scheduler_curve', 'cosine')
        if scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps
            )
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
        
    def load_model_weights(self, model, model_state_dict):
        try:
            model.load_state_dict(model_state_dict)
            print(f"Loading all model params..")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = model_state_dict
            pretrained_dict = {
                k: v
                for k, v in model_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

###################################################################################

class OncoformerOmics(LitOncoformerBase):
    """
    Multi‑omics student/teacher with interleaved self‑attn and cross‑stitch.
    - One ModalityMLM per modality (DNA, RNA, ...), built from config
    - Optional DeepCrossStitcher per layer
    - CLIP-MoCo student/teacher heads on pooled fused embedding (cross-modal contrastive)
    - Per‑modality MLM via fused wrapper for model_masking_prediction()
    """
    def __init__(self, config: Dict[str, Any], tokenizers: Dict[str, Any], checkpoint_dir: str = 'checkpoints'):
        super().__init__(config, checkpoint_dir)
        self.parent_config = cp.deepcopy(self.config)
        arch = self.config['omics']['architecture']
        self.modalities: List[str] = list(arch['modalities'])  # e.g. ['dna','rna']
        assert len(self.modalities) >= 1, "At least one modality required"

        self.shared_dim = arch['embed_dim']   # enforce same D across streams
        self.pool_embeddings = arch.get('pool_embeddings', 'cls_token')  # 'mean' | 'cls_token' | 'cls_attn'
        # Optional attention aggregation controls (compatible with DNAModel)
        self.pool_attn = arch.get('pool_attn', 'mean')
        self.pool_attn_layer = arch.get('pool_attn_layer', None)
        self.tokenizers = tokenizers
        viz_cfg = self.config.get('visualization', {})
        self.save_self_attn = bool(viz_cfg.get('save_self_attn', False))

        # save configs
        self.config_hash = self.get_config_hash()
        self.config_checkpoint_dir = os.path.join(self.checkpoint_dir, f'omics_{self.config_hash}')
        os.makedirs(self.config_checkpoint_dir, exist_ok=True)
        cfg_file = os.path.join(self.config_checkpoint_dir, 'config.yaml')
        if not os.path.exists(cfg_file):
            OmegaConf.save(config=self.config, f=cfg_file)
        parent_cfg_file = os.path.join(self.config_checkpoint_dir, 'parent_config.yaml')
        if not os.path.exists(parent_cfg_file):
            OmegaConf.save(config=self.parent_config, f=parent_cfg_file)

        self.encoder_types = {
            'LayerNormEncoder': LayerNormEncoder,
            'IntegerEncoder': IntegerEncoder,
            'FourierFractionEncoder': FourierFractionEncoder,
            'Embedding': nn.Embedding,
            'PrecomputedEmbeddingsEncoder': PrecomputedEmbeddingsEncoder,
            'PrecomputedEmbeddingsAtomicEncoder': PrecomputedEmbeddingsAtomicEncoder,
            'FourierFiLMGateEncoder': FourierFiLMGateEncoder,
        }
        
        # ---------------- Build per‑modality streams (students) ----------------
        self.streams = nn.ModuleDict()
        self.graph_cfg_by_mod: Dict[str, Dict[str, Any]] = {}
        # record components with predict=True for explanations
        self.predict_components_by_mod = {}
        for m in self.modalities:
            comp_cfg = self._build_component_configs_for(m)
            pad_idx = self._get_mod_pad_idx(m)

            graph_cfg = (
                (self.config['omics']['modalities'].get(m, {}) or {})
                .get('architecture', {})
                .get('graph', {})
                or {}
            )
            if graph_cfg.get('enabled', False):
                self.graph_cfg_by_mod[m] = {
                    'enabled': True,
                    'kernel': str(graph_cfg.get('kernel', 'diffusion')).lower(),
                    'beta': float(graph_cfg.get('beta', 0.5)),
                    'cheby_K': int(graph_cfg.get('cheby_K', 6)),
                    'preserve_cls': bool(graph_cfg.get('preserve_cls', True)),
                }
            else:
                self.graph_cfg_by_mod[m] = {'enabled': False}

            modality_encoder = ModalityEncoder(
                embedding_dim=self.shared_dim,
                pad_token_id=pad_idx,
                na_token_id=None,
                component_configs=comp_cfg,
            )
            self.streams[m] = ModalityMLM(
                modality_encoder=modality_encoder,
                component_configs=comp_cfg,
                embed_dim=self.shared_dim,
                num_heads=arch['num_heads'],
                hidden_dim=arch['hidden_dim'],
                num_layers=arch.get('num_layers', 4),
                max_length=arch.get('max_length', 8192),
                dropout=arch.get('dropout', 0.1),
                pad_idx=pad_idx,
                save_self_attn=self.save_self_attn,
            )
            # save for predict_step explanations
            self.predict_components_by_mod[m] = sorted(self.streams[m].mlm_heads.keys())

        # ---------------- Cross‑stitchers ----------------
        st = arch.get('stitch', {})
        self.stitch_enabled = st.get('enabled', True) and len(self.modalities) > 1
        if self.stitch_enabled:
            L = arch.get('num_layers', 4)
            self.stitchers = nn.ModuleList([
                DeepCrossStitcher(
                    modalities=self.modalities,
                    d_model=self.shared_dim,
                    nhead=st.get('nhead', arch['num_heads']),
                    dropout=st.get('dropout', arch.get('dropout', 0.1)),
                    pattern=st.get('pattern', 'cls2all'),
                    ff_ratio=st.get('ff_ratio', 4.0),
                    gate_init=st.get('gate_init', -2.0),
                ) for _ in range(L)
            ])
            if st.get('topk', None) is not None:
                for s in self.stitchers:
                    s.topk = st['topk']
        else:
            self.stitchers = None

        # ---------------- Teacher copies for EMA (streams and stitchers) ----------------
        self.streams_t = cp.deepcopy(self.streams)
        for p in self.streams_t.parameters():
            p.requires_grad = False
        if self.stitchers is not None:
            self.stitchers_t = cp.deepcopy(self.stitchers)
            for p in self.stitchers_t.parameters():
                p.requires_grad = False
        else:
            self.stitchers_t = None

        # ---------------- CLIP-MoCo heads and hyperparams ----------------
        self.embed_dim = self.shared_dim  # pooled fused dim
        clip_cfg = self.config['training']
        self.clip_queue_size = int(clip_cfg.get('clip_queue_size', 65536))
        self.clip_out_dim = int(clip_cfg.get('clip_out_dim', 256))
        self.clip_hidden = int(clip_cfg.get('clip_hidden_dim', 2048))
        self.clip_temperature = float(clip_cfg.get('clip_temperature', 0.07))
        self.clip_m_base = float(clip_cfg.get('clip_teacher_m_base', 0.996))
        self.clip_m_final = float(clip_cfg.get('clip_teacher_m_final', 0.9995))

        # Per-modality projection heads (student + teacher)
        self.clip_heads_q = nn.ModuleDict()
        self.clip_heads_k = nn.ModuleDict()
        for m in self.modalities:
            self.clip_heads_q[m] = CLIPMoCoHead(self.embed_dim, self.clip_hidden, self.clip_out_dim)
            self.clip_heads_k[m] = CLIPMoCoHead(self.embed_dim, self.clip_hidden, self.clip_out_dim)
            self.clip_heads_k[m].load_state_dict(self.clip_heads_q[m].state_dict(), strict=True)
            for p in self.clip_heads_k[m].parameters():
                p.requires_grad = False

        # Per-modality queues for cross-modal contrastive
        for m in self.modalities:
            queue = F.normalize(torch.randn(self.clip_out_dim, self.clip_queue_size), dim=0)
            self.register_buffer(f'{m}_queue', queue)
            self.register_buffer(f'{m}_queue_ptr', torch.zeros(1, dtype=torch.long))

        # Fused queue for single-modality fallback (standard MoCo)
        fused_queue = F.normalize(torch.randn(self.clip_out_dim, self.clip_queue_size), dim=0)
        self.register_buffer('fused_queue', fused_queue)
        self.register_buffer('fused_queue_ptr', torch.zeros(1, dtype=torch.long))

        # task noise scalars for loss balancing
        self.log_var_recon = nn.Parameter(torch.tensor(0.0))
        self.log_var_clip = nn.Parameter(torch.tensor(0.0))

        # ModDrop (for augmentation in contrastive views)
        self.moddrop_student_p = float(self.config['training'].get('moddrop_student_p', 0.25))
        self.moddrop_teacher_p = float(self.config['training'].get('moddrop_teacher_p', 0.0))

        # Optim / scheduler bookkeeping (populated at runtime)
        self._optimizer: Optional[optim.Optimizer] = None
        self._lr_scheduler_placeholder: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self._total_opt_steps: Optional[int] = None
        self._external_step: int = 0

        # EMA-tracked class frequencies for lift metrics (proportional chance baseline)
        self._class_freq_ema: Dict[Tuple[str, str], torch.Tensor] = {}
        self._freq_ema_momentum = 0.99

    # ---------------- config helpers ----------------
    def _get_mod_cfg_root(self, modality: str) -> Dict[str, Any]:
        # expected structure: config['omics']['modalities'][modality]
        try:
            return self.config['omics']['modalities'][modality]
        except Exception:
            raise KeyError(f"Missing config['omics']['modalities']['{modality}']")

    def _get_mod_pad_idx(self, modality: str) -> int:
        root = self._get_mod_cfg_root(modality)
        return int(root.get('pad_idx', self.config['omics']['architecture'].get('pad_idx', 0)))

    def _build_component_configs_for(self, modality: str) -> Dict[str, Dict[str, Any]]:
        """
        Rebuild component configs (resolve encoder types and num_embeddings).
        Expected:
          config['omics']['modalities'][modality]['architecture']['component_configs']
          config['omics']['modalities'][modality]['vocab']['file']  (optional)
        """
        root = self._get_mod_cfg_root(modality)
        comp_cfg = cp.deepcopy(root['architecture']['component_configs'])
        # optional vocab file for num_embeddings
        vocab_file = root.get('vocab', {}).get('file', None)
        vocab_sizes = {}
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            vocab_sizes = vocab.get('vocab_sizes', {})
        # resolve encoder types and set num_embeddings
        for comp_name, cfg in comp_cfg.items():
            enc_name = cfg.get('encoder_type')
            enc_cls = None
            if enc_name:
                enc_cls = self.encoder_types.get(enc_name, None)
                if enc_cls is None:
                    raise ValueError(f"[{modality}] Encoder type '{enc_name}' not found")
                cfg['encoder_type'] = enc_cls

            predict_setting = cfg.get('predict', False)

            if comp_name in vocab_sizes:
                cfg['num_embeddings'] = vocab_sizes[comp_name]
                if cfg['params'].get('num_embeddings', False):
                    cfg['params']['num_embeddings'] = cfg['num_embeddings']
                elif 'num_embeddings' in cfg['params']:
                    del cfg['params']['num_embeddings']
            else:
                if predict_setting is True:
                    params = cfg.get('params', {})
                    n_classes = cfg.get('num_embeddings', None)
                    if n_classes is None:
                        n_classes = params.get('n_bins', params.get('max_value', None))
                    if n_classes is None:
                        raise ValueError(
                            f"[{modality}:{comp_name}] predict=True but no 'num_embeddings' or 'params.n_bins/max_value'"
                        )
                    cfg['num_embeddings'] = int(n_classes)

            if enc_cls is not None and issubclass(enc_cls, PrecomputedEmbeddingsEncoder):
                params = cfg.setdefault('params', {})
                data_root = root.get('data', {})
                if params.get('weight_path', None) is None:
                    params['weight_path'] = data_root.get(f"{comp_name}_weight_path")
                if params.get('vocab_path', None) is None:
                    params['vocab_path']  = data_root.get(f"{comp_name}_vocab_path")
                if issubclass(enc_cls, PrecomputedEmbeddingsAtomicEncoder):
                    params.setdefault('trainable_base', False)
                    params.setdefault('trainable_atomic', True)
                else:
                    params.setdefault('trainable', False)

        return comp_cfg

    # ---------------- general helpers ----------------
    def _unpack_batch(self, batch: Dict[str, Any]):
        """
        Return (inputs_by_mod, masks_by_mod) in unified format.
    
        Accepts either:
          - new format: batch['omics_inputs'], batch['omics_masks'], or
          - legacy format: batch['{m}'], batch['{m}_attention_mask'] per modality.
        """
        if ('omics_inputs' in batch) and ('omics_masks' in batch):
            graphs = batch.get('omics_graphs', {})
            return batch['omics_inputs'], batch['omics_masks'], graphs
    
        inputs = {}
        masks  = {}
        for m in self.modalities:
            if m not in batch or f'{m}_attention_mask' not in batch:
                raise KeyError(f"Batch missing keys for modality '{m}' "
                               f"(expected '{m}' and '{m}_attention_mask' or omics_* dicts).")
            inputs[m] = batch[m]
            masks[m]  = batch[f'{m}_attention_mask']
            
        return inputs, masks, {}

    # ---------------- interleaved encoders ----------------
    def _encode_interleaved_with(self, streams: nn.ModuleDict, stitchers: Optional[nn.ModuleList],
                                 inputs: Dict[str, Dict[str, torch.Tensor]],
                                 masks: Dict[str, torch.Tensor],
                                 graphs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        Returns final token embeddings per modality after L blocks of:
            self-attn (per modality) -> cross-stitch (all pairs)
        """
        graphs = graphs or {}
        # initial embeddings
        H = {m: streams[m].initial_embed(inputs[m]) for m in self.modalities}
        L = len(streams[self.modalities[0]].layers)
        for l in range(L):
            # self-attention per modality (one layer)
            for m in self.modalities:
                graph_data = graphs.get(m, None)
                graph_cfg = getattr(self, 'graph_cfg_by_mod', {}).get(m, None)
                H[m] = streams[m].encode_tokens_step(
                    H[m],
                    masks[m],
                    l,
                    graph_data=graph_data,
                    graph_cfg=graph_cfg,
                )
            # compute salience from layer attention (CLS→token mean over heads)
            sal = {}
            for m in self.modalities:
                a = streams[m].attn[l].data  # [B, H, Lq, Lk]
                if a is not None:
                    sal[m] = a.mean(dim=1)[:, 0, :]  # [B, Lk]
                else:
                    sal[m] = None
            # cross‑stitch
            if stitchers is not None:
                H = stitchers[l](H, masks, salience=sal)
                
        return H

    def _encode_interleaved(self, inputs: Dict[str, Dict[str, torch.Tensor]],
                            masks: Dict[str, torch.Tensor],
                            graphs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        return self._encode_interleaved_with(self.streams, self.stitchers, inputs, masks, graphs)

    @torch.no_grad()
    def _encode_interleaved_teacher(self, inputs: Dict[str, Dict[str, torch.Tensor]],
                                    masks: Dict[str, torch.Tensor],
                                    graphs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        return self._encode_interleaved_with(self.streams_t, self.stitchers_t, inputs, masks, graphs)

    # ---------------- pooling ----------------
    def _pool(self, H: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.pool_embeddings == 'cls_token':
            # Weight CLS by modality presence (mask[:,0]) and average over present modalities only
            cls_list = []
            w_list = []
            for m in self.modalities:
                cls_list.append(H[m][:, 0, :])                        # [B, D]
                w_list.append(masks[m][:, 0].float().unsqueeze(-1))   # [B, 1]
            C = torch.stack(cls_list, dim=0)  # [M, B, D]
            W = torch.stack(w_list, dim=0)    # [M, B, 1]
            num = (C * W).sum(dim=0)          # [B, D]
            denom = W.sum(dim=0).clamp_min(1e-9)  # [B,1]
            pooled = num / denom
            # Fallback when all masks are zero: simple mean across modalities
            all_zero = (W.sum(dim=0).squeeze(-1) <= 0)
            if all_zero.any():
                fallback = C.mean(dim=0)
                pooled[all_zero] = fallback[all_zero]
            return pooled
        elif self.pool_embeddings == 'mean':
            means = []
            for m in self.modalities:
                w = masks[m].float().unsqueeze(-1)
                means.append((H[m]*w).sum(1) / (w.sum(1)+1e-9))
            return torch.stack(means, dim=0).mean(dim=0)
        elif self.pool_embeddings == 'cls_attn':
            # attention-weighted token pooling using aggregated self-attention
            pooled_list = []
            for m in self.modalities:
                A = self._avg_self_attn(m)  # [B, Lq, Lk] or None
                if A is None:
                    pooled_list.append(H[m][:, 0, :])
                    continue
                cls_to_tok = A[:, 0, :]                  # [B, L]
                wmask = masks[m].float()                 # [B, L]
                w = cls_to_tok * wmask                   # include CLS weight like DNAModel
                denom = w.sum(dim=1, keepdim=True)
                norm = torch.where(denom > 0, w / (denom + 1e-9), torch.zeros_like(w))
                pooled_m = (H[m] * norm.unsqueeze(-1)).sum(dim=1)
                bad = (denom.squeeze(1) <= 0)
                if bad.any():
                    pooled_m[bad] = H[m][bad, 0, :]
                pooled_list.append(pooled_m)
            return torch.stack(pooled_list, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown pool_embeddings '{self.pool_embeddings}'")

    def _avg_self_attn(self, modality: str) -> Optional[torch.Tensor]:
        """Aggregate self-attention across layers (and heads) similar to DNAModel.
        Returns [B, Lq, Lk] or None if no attention is available.
        """
        hooks = self.streams[modality].attn
        if not hooks:
            return None
        if self.pool_attn_layer is None:
            layers = [h.data for h in hooks if h.data is not None]
            if not layers:
                return None
            A = torch.stack(layers, dim=0).mean(dim=(0, 2))  # [B, Lq, Lk]
            if isinstance(self.pool_attn, str) and self.pool_attn.lower() == 'max':
                A = torch.stack([h.data.mean(dim=1) for h in hooks if h.data is not None], dim=0).max(dim=0).values
        else:
            idx = int(self.pool_attn_layer)
            if idx < 0 or idx >= len(hooks) or hooks[idx].data is None:
                return None
            A = hooks[idx].data.mean(dim=1)  # [B, Lq, Lk]
        return A

    def _pool_single(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool_embeddings == 'cls_token':
            cls_vec = hidden[:, 0, :]
            mask_cls = mask[:, 0].float()
            pooled = cls_vec
            missing = mask_cls <= 0
            if missing.any():
                w = mask.float().unsqueeze(-1)
                denom = w.sum(1, keepdim=True).clamp_min(1e-9)
                mean_vec = (hidden * w).sum(1) / denom
                pooled = pooled.clone()
                pooled[missing] = mean_vec[missing]
            return pooled
        elif self.pool_embeddings == 'mean':
            w = mask.float().unsqueeze(-1)
            denom = w.sum(1, keepdim=True).clamp_min(1e-9)
            return (hidden * w).sum(1) / denom
        elif self.pool_embeddings == 'cls_attn':
            cls_vec = hidden[:, 0, :]
            return cls_vec
        else:
            raise ValueError(f"Unknown pool_embeddings '{self.pool_embeddings}'")

    # ---------------- CLIP-MoCo helpers ----------------
    def _current_opt_step(self) -> int:
        if self.trainer is not None:
            return int(self.global_step)
        return int(getattr(self, "_external_step", 0))

    def _total_steps_fallback(self) -> int:
        training_cfg = self.config.get('training', {})
        training_len = int(training_cfg.get('training_len', 15000))
        num_epochs = int(training_cfg.get('num_epochs', 20))
        return max(training_len * num_epochs, 1)

    def _keep_ratio_for(self, m: str) -> float:
        """Keep ratio for token masking in contrastive views."""
        return max(
            0.05,
            1.0 - float(self.config['training'].get(
                f'{m}_contrast_mask_rate',
                self.config['training'].get('dna_contrast_mask_rate', 0.25)
            ))
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue_name: str, keys: torch.Tensor):
        """Update the queue with new keys (teacher embeddings)."""
        # DDP: gather keys from every rank so all ranks enqueue the same, full set
        # of negatives (standard MoCo-v2 behavior). Without this, each of the 4 GPUs
        # keeps an independent queue seeing only 1/4 of the batch. No-op on 1 GPU.
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            keys = keys.contiguous()
            gathered = [torch.empty_like(keys) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, keys)
            keys = torch.cat(gathered, dim=0)
        queue = getattr(self, queue_name)
        ptr_name = f'{queue_name}_ptr'
        ptr_buf = getattr(self, ptr_name)
        batch_size = keys.size(0)
        queue_size = queue.size(1)
        ptr = int(ptr_buf)

        if ptr + batch_size <= queue_size:
            queue[:, ptr:ptr + batch_size] = keys.T
        else:
            rem = queue_size - ptr
            queue[:, ptr:] = keys[:rem].T
            queue[:, :batch_size - rem] = keys[rem:].T
        ptr_buf[0] = (ptr + batch_size) % queue_size

    def _clip_moco_loss(self, batch: Dict[str, Any], return_metrics: bool = False):
        """
        CLIP-MoCo cross-modal contrastive loss with queue-based negatives.
        
        Multi-modality (2+): Cross-modal InfoNCE (DNA↔RNA) with all-pairs.
        Single-modality: Falls back to standard MoCo with augmented views.
        """
        inputs_by_mod, masks_by_mod, graphs_by_mod = self._unpack_batch(batch)

        # Determine which modalities have data in this batch
        present_mods = [m for m in self.modalities if masks_by_mod[m].sum() > 0]

        if len(present_mods) >= 2:
            # Cross-modal CLIP-MoCo: all pairwise combinations
            return self._cross_modal_clip_loss(
                inputs_by_mod, masks_by_mod, graphs_by_mod, present_mods, return_metrics
            )
        elif len(present_mods) == 1:
            # Single-modality fallback: standard MoCo with augmented views
            return self._same_mod_moco_loss(
                inputs_by_mod, masks_by_mod, graphs_by_mod, present_mods[0], return_metrics
            )
        else:
            # No modalities present
            zero = torch.tensor(0.0, device=self.device)
            if return_metrics:
                return zero, {'clip_loss': 0.0, 'clip_pairs': 0, 'clip_mode': 'none'}
            return zero

    def _cross_modal_clip_loss(
        self,
        inputs_by_mod: Dict[str, Dict[str, torch.Tensor]],
        masks_by_mod: Dict[str, torch.Tensor],
        graphs_by_mod: Optional[Dict[str, Dict[str, Any]]],
        present_mods: List[str],
        return_metrics: bool = False,
    ):
        """All-pairs cross-modal InfoNCE with queue negatives."""
        # Encode with student and teacher using UNIMODAL encoding (no cross-stitch)
        # to prevent information leakage between modalities during contrastive learning
        H_s = self._encode_interleaved_with(
            self.streams, None, inputs_by_mod, masks_by_mod, graphs_by_mod
        )
        with torch.no_grad():
            H_t = self._encode_interleaved_with(
                self.streams_t, None, inputs_by_mod, masks_by_mod, graphs_by_mod
            )

        loss = torch.tensor(0.0, device=self.device)
        pairs = 0
        per_sample_present = {m: masks_by_mod[m].sum(dim=1) > 0 for m in present_mods}

        for src in present_mods:
            for tgt in present_mods:
                if src == tgt:
                    continue

                # Find samples that have both modalities
                valid = per_sample_present[src] & per_sample_present[tgt]
                if not valid.any():
                    continue

                # Student query from src modality
                src_pool = self._pool_single(H_s[src], masks_by_mod[src])
                q = self.clip_heads_q[src](src_pool)[valid]

                # Teacher key from tgt modality
                with torch.no_grad():
                    tgt_pool = self._pool_single(H_t[tgt], masks_by_mod[tgt])
                    k = self.clip_heads_k[tgt](tgt_pool)[valid]

                # InfoNCE: positive = q·k, negatives = q·queue
                l_pos = (q * k).sum(dim=-1, keepdim=True)  # [N, 1]
                tgt_queue = getattr(self, f'{tgt}_queue').clone().detach()
                l_neg = q @ tgt_queue  # [N, queue_size]
                logits = torch.cat([l_pos, l_neg], dim=1) / self.clip_temperature
                labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
                loss = loss + F.cross_entropy(logits, labels)
                pairs += 1

                # Enqueue teacher keys for this target modality
                self._dequeue_and_enqueue(f'{tgt}_queue', k)

        if pairs > 0:
            loss = loss / pairs

        if return_metrics:
            return loss, self._clip_metrics(loss, pairs, mode='cross_mod')
        return loss

    def _same_mod_moco_loss(
        self,
        inputs_by_mod: Dict[str, Dict[str, torch.Tensor]],
        masks_by_mod: Dict[str, torch.Tensor],
        graphs_by_mod: Optional[Dict[str, Dict[str, Any]]],
        mod: str,
        return_metrics: bool = False,
    ):
        """Single-modality MoCo fallback: two augmented views of same data."""
        device = self.device

        # Create two augmented views via token masking + moddrop
        masks_v1 = {m: _build_keep_mask(masks_by_mod[m], keep_ratio=self._keep_ratio_for(m))
                    for m in self.modalities}
        masks_v2 = {m: _build_keep_mask(masks_by_mod[m], keep_ratio=self._keep_ratio_for(m))
                    for m in self.modalities}

        masks_v1 = _maybe_moddrop(masks_v1, p_drop=self.moddrop_student_p)
        masks_v2 = _maybe_moddrop(masks_v2, p_drop=self.moddrop_teacher_p)

        # Optional VAF augmentation for DNA
        inputs_v1 = {m: {k: v.clone() for k, v in d.items()} for m, d in inputs_by_mod.items()}
        inputs_v2 = {m: {k: v.clone() for k, v in d.items()} for m, d in inputs_by_mod.items()}

        if 'dna' in inputs_by_mod and 'aa_vaf' in inputs_by_mod['dna']:
            tcfg = self.config.get("training", {})
            epoch = getattr(self, "current_epoch", 0)
            warm = tcfg.get("dna_contrast_aug_warm_epochs", 3)
            noise_hi = tcfg.get("dna_contrast_vaf_noise_hi", 0.006)
            noise_lo = tcfg.get("dna_contrast_vaf_noise_lo", 0.003)
            eps_abs = noise_hi if epoch < warm else noise_lo
            a_lo = tcfg.get("dna_contrast_vaf_a_lo", 0.97)
            a_hi = tcfg.get("dna_contrast_vaf_a_hi", 1.03)
            b_abs = tcfg.get("dna_contrast_vaf_b_abs", 0.01)

            def _vaf_aug(x, valid_mask):
                if valid_mask is None:
                    valid_mask = torch.ones_like(x, dtype=torch.bool)
                B = x.size(0)
                a = (a_lo + (a_hi - a_lo) * torch.rand(B, device=x.device)).view(B, 1)
                b = (2 * torch.rand(B, device=x.device) - 1.0).view(B, 1) * b_abs
                eps = (2 * torch.rand_like(x) - 1.0) * eps_abs
                y = a * x + b + eps
                return torch.where(valid_mask, y, x).clamp(0.0, 1.0)

            v1_mask = masks_v1['dna'].bool() if 'dna' in masks_v1 else None
            v2_mask = masks_v2['dna'].bool() if 'dna' in masks_v2 else None
            inputs_v1['dna']['aa_vaf'] = _vaf_aug(inputs_v1['dna']['aa_vaf'].to(device), v1_mask)
            inputs_v2['dna']['aa_vaf'] = _vaf_aug(inputs_v2['dna']['aa_vaf'].to(device), v2_mask)

        # Encode both views using UNIMODAL encoding (no cross-stitch)
        # to prevent information leakage during contrastive learning
        H_s = self._encode_interleaved_with(
            self.streams, None, inputs_v1, masks_v1, graphs_by_mod
        )
        z1 = self._pool(H_s, masks_v1)
        with torch.no_grad():
            H_t = self._encode_interleaved_with(
                self.streams_t, None, inputs_v2, masks_v2, graphs_by_mod
            )
            z2 = self._pool(H_t, masks_v2)

        # Project through the single modality's head
        q = self.clip_heads_q[mod](z1)
        with torch.no_grad():
            k = self.clip_heads_k[mod](z2)

        # InfoNCE with fused queue
        l_pos = (q * k).sum(dim=-1, keepdim=True)
        l_neg = q @ self.fused_queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.clip_temperature
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)

        # Enqueue to fused queue
        self._dequeue_and_enqueue('fused_queue', k)

        if return_metrics:
            return loss, self._clip_metrics(loss, 1, mode='same_mod', same_mod=mod)
        return loss

    def _clip_metrics(self, loss, pairs, mode='cross_mod', same_mod=None):
        """Build metrics dictionary for CLIP-MoCo logging."""
        metrics = {
            'clip_loss': float(loss.detach()) if torch.is_tensor(loss) else loss,
            'clip_pairs': pairs,
            'clip_temperature': self.clip_temperature,
            'clip_mode_is_cross': 1.0 if mode == 'cross_mod' else 0.0,  # Numeric for logging
        }
        # Store string mode for internal use but not for logging
        metrics['clip_mode'] = mode  # Will be filtered out before log_dict
        if same_mod:
            metrics['clip_same_mod'] = same_mod  # Will be filtered out before log_dict
        # Add queue utilization metrics
        for m in self.modalities:
            ptr = int(getattr(self, f'{m}_queue_ptr'))
            metrics[f'clip_{m}_queue_ptr'] = ptr
        metrics['clip_fused_queue_ptr'] = int(self.fused_queue_ptr)
        return metrics

    def _compute_lift(self, mod: str, comp: str, acc: float) -> float:
        """
        Compute lift = accuracy / proportional_chance_baseline.
        
        Proportional chance = Σ(p_i²) where p_i is the frequency of class i.
        This is the expected accuracy of a model predicting according to marginal distribution.
        Lift > 1.0 means the model beats weighted random guessing.
        """
        key = (mod, comp)
        freq = self._class_freq_ema.get(key)
        if freq is None or acc <= 0:
            return 0.0
        proportional_chance = (freq ** 2).sum().item()  # Σ(p_i²)
        return acc / proportional_chance if proportional_chance > 1e-9 else 0.0

    # ---------------- MLM via fused forward ----------------
    def _mlm_loss_mm(
        self,
        target_modality: str,
        inputs_by_mod: Dict[str, Dict[str, torch.Tensor]],
        masks_by_mod: Dict[str, torch.Tensor],
        graphs_by_mod: Optional[Dict[str, Dict[str, Any]]] = None,
        mlm_probability: float = 0.15,
        mask_whole_alterations: bool = True,
        mask_whole_components: bool = False,
        context_moddrop_p: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Fused MLM loss for ONE target modality using other modalities as context (with optional ModDrop).
        """
        device = self.device
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01, ignore_index=-100, reduction='none')
    
        # 1) Mask TARGET modality only (token-level), leave context untouched
        t_inputs = inputs_by_mod[target_modality]
        t_tok    = self.tokenizers[target_modality]
        t_stream = self.streams[target_modality]
        t_inputs_masked, t_labels = mask_tokens(
            t_inputs, t_tok, device,
            mlm_probability=mlm_probability,
            mask_whole_alterations=mask_whole_alterations,
            mask_whole_components=mask_whole_components,
            encoded_components=list(t_stream.modality_encoder.encoders.keys()),
            predict_heads=list(t_stream.mlm_heads.keys()),
            predict_alias=getattr(t_stream, "predict_alias_map", {}),
            float_pad_values=getattr(t_tok, "component_pad_values", {}),
        )
    
        masked_inputs = {m: (t_inputs_masked if m == target_modality else inputs_by_mod[m])
                         for m in self.modalities}
    
        # 2) Apply modality drop to CONTEXT masks (never drop the target here)
        ctx = {m: masks_by_mod[m] for m in self.modalities if m != target_modality}
        ctx_dropped = _maybe_moddrop(ctx, p_drop=context_moddrop_p)
        fused_masks = {m: (masks_by_mod[m] if m == target_modality else ctx_dropped[m])
                       for m in self.modalities}
    
        # 3) Interleaved forward once for all modalities
        H = self._encode_interleaved(masked_inputs, fused_masks, graphs_by_mod)
    
        # 4) Apply MLM heads for the TARGET modality and compute CE only on masked positions
        logits_dict = self.streams[target_modality].apply_mlm_heads(H[target_modality])
    
        per_comp_losses = []
        per_comp_acc = {}
    
        for comp_name, logits in logits_dict.items():
            labels = t_labels[comp_name].to(device).view(-1)             # [B*L]
            logits2d = logits.view(-1, logits.size(-1))                  # [B*L, V]
            loss_vec = criterion(logits2d, labels)                       # [B*L]
            weight   = self.streams[target_modality].modality_encoder.loss_param.get(comp_name, 1.0)
            per_comp_losses.append(loss_vec * weight)
    
            # accuracy on masked positions
            preds = logits.argmax(dim=-1)
            valid = (t_labels[comp_name] != -100)
            if valid.any():
                acc = (preds[valid] == t_labels[comp_name][valid]).float().mean().item()

                # Update EMA class frequencies for lift computation
                valid_labels = t_labels[comp_name][valid].view(-1)
                vocab_size = logits.size(-1)
                counts = torch.bincount(valid_labels, minlength=vocab_size).float()
                batch_freq = counts / counts.sum().clamp_min(1)

                key = (target_modality, comp_name)
                if key not in self._class_freq_ema:
                    self._class_freq_ema[key] = batch_freq.detach().cpu()
                else:
                    self._class_freq_ema[key] = (
                        self._freq_ema_momentum * self._class_freq_ema[key]
                        + (1 - self._freq_ema_momentum) * batch_freq.detach().cpu()
                    )
            else:
                acc = 0.0
            per_comp_acc[comp_name] = acc
    
        loss = finite_reduce_loss(per_comp_losses)
        return loss, per_comp_acc

    def fused_mlm_forward(
        self,
        target_modality: str,
        inputs_by_mod: Dict[str, Dict[str, torch.Tensor]],
        masks_by_mod: Dict[str, torch.Tensor],
        graphs_by_mod: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the shared interleaved fusion once, then emit MLM logits for the target modality.
        """
        H = self._encode_interleaved(inputs_by_mod, masks_by_mod, graphs_by_mod)
        logits = {comp: head(H[target_modality]) 
                  for comp, head in self.streams[target_modality].mlm_heads.items()}
        
        return logits
    
    def fused_encode_student(self, inputs_by_mod, masks_by_mod, graphs_by_mod=None) -> torch.Tensor:
        H = self._encode_interleaved(inputs_by_mod, masks_by_mod, graphs_by_mod)
        return self._pool(H, masks_by_mod)
    
    @torch.no_grad()
    def fused_encode_teacher(self, inputs_by_mod, masks_by_mod, graphs_by_mod=None) -> torch.Tensor:
        H = self._encode_interleaved_teacher(inputs_by_mod, masks_by_mod, graphs_by_mod)
        return self._pool(H, masks_by_mod)
        
    # ---------------- Lightning API ----------------
    def calculate_embeddings(self, batch: Dict[str, Any]):
        inputs, masks, graphs = self._unpack_batch(batch)
        return self._encode_interleaved(inputs, masks, graphs)

    def calculate_pooled_embedding(self, batch: Dict[str, Any]):
        inputs, masks, graphs = self._unpack_batch(batch)
        H = self._encode_interleaved(inputs, masks, graphs)
        return self._pool(H, masks)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._center_buf = []

        if self.trainer is None:
            self._total_opt_steps = self._total_steps_fallback()
            return

        total_opt_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        self._total_opt_steps = max(total_opt_steps, 1)

        training_cfg = self.config.setdefault('training', {})
        warmup_steps_cfg = training_cfg.get('scheduler_warmup_steps')
        if warmup_steps_cfg is None:
            warmup_fraction = float(training_cfg.get('scheduler_warmup_fraction', 0.1))
            warmup_steps = int(self._total_opt_steps * warmup_fraction)
            training_cfg['scheduler_warmup_steps'] = warmup_steps
        else:
            warmup_steps = int(warmup_steps_cfg)

        scheduler_type = training_cfg.get('scheduler_curve', 'cosine')
        optimizer = self._optimizer
        if optimizer is None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            self._optimizer = optimizer
        if optimizer is None:
            return

        if scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self._total_opt_steps,
            )
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self._total_opt_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self._scheduler = scheduler

        configs = getattr(self.trainer.strategy, "_lr_scheduler_configs", None)
        if configs:
            configs[0].scheduler = scheduler
            configs[0].interval = "step"
            configs[0].frequency = 1

        self._lr_scheduler_placeholder = scheduler

    @torch.no_grad()
    def predict_step(self, batch: Dict[str, Any], batch_idx: Optional[int] = None):
        inputs, masks, graphs = self._unpack_batch(batch)
        toggled_streams: List[ModalityMLM] = []
        try:
            for m in self.modalities:
                stream = self.streams[m]
                if hasattr(stream, "clear_attention_buffers"):
                    stream.clear_attention_buffers()
            if not self.save_self_attn:
                for m in self.modalities:
                    stream = self.streams[m]
                    if getattr(stream, "attention_capture_enabled", False):
                        continue
                    stream.enable_attention_capture()
                    toggled_streams.append(stream)

            H = self._encode_interleaved(inputs, masks, graphs)
            pooled = self._pool(H, masks)

            sample_metadata = batch['sample_metadata']['sample_metadata']

            # ---- build attention explanations (self + cross) ----
            attn_expl = {"self": {}, "cross": {}}

            def _topk_from_attn(attn_4d, mask_2d, k: int):
                if attn_4d is None:
                    return None, None
                a = attn_4d.mean(dim=1)      # [B,Lq,Lk]
                if a.size(1) > 1:
                    a = a.max(dim=1).values  # [B,Lk]
                else:
                    a = a[:, 0, :]
                mask = mask_2d.bool().clone()
                if mask.size(1) > 0:
                    # exclude CLS from selection
                    mask[:, 0] = False
                a = a.masked_fill(~mask, float('-inf'))
                if a.numel() == 0:
                    return None, None
                kk = max(1, min(int(k), a.size(1)))
                scores, idx = torch.topk(a, k=kk, dim=1, largest=True)
                # Ensure at least a valid index when everything is -inf
                all_neg_inf = torch.isinf(scores[:, 0]) & (scores[:, 0] < 0)
                if all_neg_inf.any():
                    idx = idx.clone()
                    # try first non-CLS valid token as fallback; if none, keep 0 (will be invalidated downstream)
                    if mask.size(1) > 1:
                        has_valid = mask[:, 1:].any(dim=1)
                        first_valid = mask[:, 1:].float().argmax(dim=1) + 1
                        fallback = torch.where(has_valid, first_valid, torch.zeros_like(first_valid))
                        idx[all_neg_inf, 0] = fallback[all_neg_inf]
                return idx, scores

            # Determine visualization Top-K (default 3)
            try:
                K_vis = int(self.config.get('visualization', {}).get('attn_topk', 3))
            except Exception:
                K_vis = 3

            # SELF: aggregated self-attn per modality (layers aggregated like DNAModel)
            for m in self.modalities:
                attn_entry = {
                    "topk": {
                        "indices": torch.empty(0, dtype=torch.long),
                        "scores": torch.empty(0),
                        "components": {},
                    },
                }
                A = self._avg_self_attn(m)
                if A is None:
                    attn_expl["self"][m] = attn_entry
                    continue
                # Also compute Top-K
                topk_idx, topk_sc = _topk_from_attn(A.unsqueeze(1), masks[m], k=K_vis)
                if topk_idx is None:
                    attn_expl["self"][m] = attn_entry
                    continue
                per_comp = {}
                per_comp_k = {}
                for comp in inputs[m].keys():
                    if comp not in inputs[m]:
                        continue
                    if topk_idx is not None:
                        tok = self.tokenizers.get(m, None)
                        vals_k = torch.gather(inputs[m][comp], dim=1, index=topk_idx)
                        # validity of selected positions: true token and not CLS (pos 0)
                        valid_k = torch.gather(masks[m].bool(), dim=1, index=topk_idx) & (topk_idx != 0)
                        if vals_k.dtype.is_floating_point:
                            vals_k = vals_k.masked_fill(~valid_k, float('nan'))
                            entry_k = {"values": vals_k}
                        else:
                            entry_k = {"ids": vals_k}
                        if tok is not None and hasattr(tok, "metadata_idx2token") and comp in tok.metadata_idx2token:
                            i2t = tok.metadata_idx2token[comp]
                            if not vals_k.dtype.is_floating_point:
                                toks_2d = [[i2t.get(int(x), "<unk>") for x in row] for row in vals_k.tolist()]
                                invalid_2d = (~valid_k).tolist()
                                for i_row, inv_row in enumerate(invalid_2d):
                                    for j_col, inv in enumerate(inv_row):
                                        if inv:
                                            toks_2d[i_row][j_col] = None
                                entry_k["tokens"] = toks_2d
                        per_comp_k[comp] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in entry_k.items()}
                attn_entry["topk"]["indices"] = topk_idx.detach().cpu()
                attn_entry["topk"]["scores"] = topk_sc.detach().cpu()
                attn_entry["topk"]["components"] = per_comp_k
                attn_expl["self"][m] = attn_entry

            # CROSS: last layer cross-stitch attention blocks (unchanged)
            if self.stitchers is not None and len(self.stitchers) > 0:
                st_last = self.stitchers[-1]
                for tgt in self.modalities:
                    for src in self.modalities:
                        if src == tgt:
                            continue
                        key = f"{tgt}<-{src}"
                        block = st_last.blocks[key] if (key in st_last.blocks) else None
                        if block is None or block.last_attn is None:
                            continue
                        # Compute Top-K for cross attention only
                        topk_idx, topk_sc = _topk_from_attn(block.last_attn, masks[src], k=K_vis)
                        if topk_idx is None:
                            continue
                        per_comp = {}
                        per_comp_k = {}
                        for comp in inputs[src].keys():
                            if comp not in inputs[src]:
                                continue
                            if topk_idx is not None:
                                tok = self.tokenizers.get(src, None)
                                vals_k = torch.gather(inputs[src][comp], dim=1, index=topk_idx)
                                valid_k = torch.gather(masks[src].bool(), dim=1, index=topk_idx) & (topk_idx != 0)
                                if vals_k.dtype.is_floating_point:
                                    vals_k = vals_k.masked_fill(~valid_k, float('nan'))
                                    entry_k = {"values": vals_k}
                                else:
                                    entry_k = {"ids": vals_k}
                                if tok is not None and hasattr(tok, "metadata_idx2token") and comp in tok.metadata_idx2token:
                                    i2t = tok.metadata_idx2token[comp]
                                    if not vals_k.dtype.is_floating_point:
                                        toks_2d = [[i2t.get(int(x), "<unk>") for x in row] for row in vals_k.tolist()]
                                        invalid_2d = (~valid_k).tolist()
                                        for i_row, inv_row in enumerate(invalid_2d):
                                            for j_col, inv in enumerate(inv_row):
                                                if inv:
                                                    toks_2d[i_row][j_col] = None
                                        entry_k["tokens"] = toks_2d
                                per_comp_k[comp] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in entry_k.items()}
                        attn_expl["cross"][key] = {
                            "topk": {
                                "indices": topk_idx.detach().cpu(),
                                "scores": topk_sc.detach().cpu(),
                                "components": per_comp_k,
                            },
                        }

            return {
                'sample_metadata': sample_metadata,
                'sample_embeddings': pooled,
                'attn_explanations': attn_expl,
            }
        finally:
            for stream in toggled_streams:
                stream.disable_attention_capture()
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int, log: bool = True):
        inputs_by_mod, masks_by_mod, graphs_by_mod = self._unpack_batch(batch)

        #print(f"{batch_idx} inside OncoformerOmics training_step")
        #print(f"{inputs_by_mod} unpacked inputs inside OncoformerOmics training_step")
        
        # --- fused MLM over chosen targets ---
        targets = self.config['training'].get('mlm_targets', self.modalities)
        context_drop = float(self.config['training'].get('context_moddrop_p', 0.0))
        mlm_losses = []
        per_mod_accs: Dict[str, Dict[str, float]] = {}
        for t in targets:
            modes_cfg = self.config['training'].get(f'{t}_mask_modes', ['alterations'])
            modes: List[str] = list(modes_cfg if isinstance(modes_cfg, (list, tuple)) else [modes_cfg])
            weights_cfg = self.config['training'].get(f'{t}_mask_weights', [1.0] * len(modes))
            weights: List[float] = list(weights_cfg if isinstance(weights_cfg, (list, tuple)) else [weights_cfg])
            if len(weights) < len(modes):
                weights.extend([1.0] * (len(modes) - len(weights)))
            rate = float(self.config['training'].get(f'{t}_masking_rate', 0.25))

            loss_parts: List[torch.Tensor] = []
            acc_sum: Dict[str, float] = {}
            acc_weight: Dict[str, float] = {}

            for mode, weight in zip(modes, weights):
                mode = str(mode).lower()
                weight = float(weight)
                if weight == 0.0:
                    continue
                mask_whole_alterations = (mode == 'alterations')
                mask_whole_components = (mode == 'components')
                l_part, accs = self._mlm_loss_mm(
                    target_modality=t,
                    inputs_by_mod=inputs_by_mod,
                    masks_by_mod=masks_by_mod,
                    graphs_by_mod=graphs_by_mod,
                    mlm_probability=rate,
                    mask_whole_alterations=mask_whole_alterations,
                    mask_whole_components=mask_whole_components,
                    context_moddrop_p=context_drop,
                )
                loss_parts.append(weight * l_part)
                for comp_name, acc in accs.items():
                    acc_sum[comp_name] = acc_sum.get(comp_name, 0.0) + weight * float(acc)
                    acc_weight[comp_name] = acc_weight.get(comp_name, 0.0) + weight

            if not loss_parts:
                continue

            l_t = sum(loss_parts)
            mlm_losses.append(l_t)
            per_mod_accs[t] = {
                comp: (acc_sum[comp] / acc_weight[comp]) if acc_weight[comp] > 0 else 0.0
                for comp in acc_sum.keys()
            }

            if log:
                self.log(f'train_{t}_mlm', l_t.detach(), prog_bar=True, on_step=True, on_epoch=True)
                for comp_name, acc in per_mod_accs[t].items():
                    self.log(
                        f'embed_train_{t}_{comp_name}_mask_accuracy',
                        acc,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    # Log lift (multiple over proportional chance baseline)
                    lift = self._compute_lift(t, comp_name, acc)
                    self.log(
                        f'embed_train_{t}_{comp_name}_mask_lift',
                        lift,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                    )
            
        if mlm_losses:
            loss_recon = finite_reduce_loss(mlm_losses)
        else:
            loss_recon = torch.tensor(0.0, device=self.device, dtype=self.log_var_recon.dtype)

        # --- CLIP-MoCo cross-modal contrastive loss ---
        loss_clip, clip_metrics = self._clip_moco_loss(batch, return_metrics=True)

        # heteroscedastic weighting
        lambda_recon = float(self.config['training'].get('recon_meta_w',
                            self.config['training'].get('dna_recon_meta_weight', 1.0)))
        lambda_clip = float(self.config['training'].get('clip_meta_w',
                            self.config['training'].get('w_clip', 0.5)))

        prec_recon = torch.exp(-self.log_var_recon)
        prec_clip = torch.exp(-self.log_var_clip)

        loss = lambda_recon * (prec_recon * loss_recon + self.log_var_recon) \
             + lambda_clip * (prec_clip * loss_clip + self.log_var_clip)

        log_dict = {
            'train_loss': loss.detach(),
            'train_recon_loss': loss_recon.detach(),
            'train_clip_loss': loss_clip.detach(),
        }
        # per-component mask accuracies with clear names
        for t, accs in per_mod_accs.items():
            for comp_name, acc in accs.items():
                log_dict[f'train_mask_acc__{t}__{comp_name}'] = float(acc)
        for k, v in clip_metrics.items():
            # Only log numeric values (skip strings like 'clip_mode' and 'clip_same_mod')
            if isinstance(v, (int, float, torch.Tensor)):
                log_dict[f'train_{k}'] = v

        accumulate = 1
        if self.trainer is not None:
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        loss_to_return = loss / accumulate if accumulate > 1 else loss

        if log:
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_to_return
        else:
            return loss_to_return, log_dict


    def validation_step(self, batch: Dict[str, Any], batch_idx: int, log: bool = True):
        inputs_by_mod, masks_by_mod, graphs_by_mod = self._unpack_batch(batch)
    
        targets = self.config['training'].get('mlm_targets', self.modalities)
        context_drop = float(self.config['training'].get('val_context_moddrop_p',
                             self.config['training'].get('context_moddrop_p', 0.0)))
        mlm_losses = []
        per_mod_accs: Dict[str, Dict[str, float]] = {}
        for t in targets:
            modes_cfg = self.config['training'].get(
                f'{t}_val_mask_modes',
                self.config['training'].get(f'{t}_mask_modes', ['alterations'])
            )
            modes: List[str] = list(modes_cfg if isinstance(modes_cfg, (list, tuple)) else [modes_cfg])
            weights_cfg = self.config['training'].get(
                f'{t}_val_mask_weights',
                self.config['training'].get(f'{t}_mask_weights', [1.0] * len(modes))
            )
            weights: List[float] = list(weights_cfg if isinstance(weights_cfg, (list, tuple)) else [weights_cfg])
            if len(weights) < len(modes):
                weights.extend([1.0] * (len(modes) - len(weights)))
            rate = float(self.config['training'].get(f'{t}_val_mask_rate',
                         self.config['training'].get(f'{t}_masking_rate', 0.25)))

            loss_parts: List[torch.Tensor] = []
            acc_sum: Dict[str, float] = {}
            acc_weight: Dict[str, float] = {}

            for mode, weight in zip(modes, weights):
                mode = str(mode).lower()
                weight = float(weight)
                if weight == 0.0:
                    continue
                mask_whole_alterations = (mode == 'alterations')
                mask_whole_components = (mode == 'components')
                l_part, accs = self._mlm_loss_mm(
                    target_modality=t,
                    inputs_by_mod=inputs_by_mod,
                    masks_by_mod=masks_by_mod,
                    graphs_by_mod=graphs_by_mod,
                    mlm_probability=rate,
                    mask_whole_alterations=mask_whole_alterations,
                    mask_whole_components=mask_whole_components,
                    context_moddrop_p=context_drop,
                )
                loss_parts.append(weight * l_part)
                for comp_name, acc in accs.items():
                    acc_sum[comp_name] = acc_sum.get(comp_name, 0.0) + weight * float(acc)
                    acc_weight[comp_name] = acc_weight.get(comp_name, 0.0) + weight

            if not loss_parts:
                continue

            l_t = sum(loss_parts)
            mlm_losses.append(l_t)
            per_mod_accs[t] = {
                comp: (acc_sum[comp] / acc_weight[comp]) if acc_weight[comp] > 0 else 0.0
                for comp in acc_sum.keys()
            }
            if log:
                self.log(f'val_{t}_mlm', l_t.detach(), prog_bar=True, on_step=False, on_epoch=True)
                for comp_name, acc in per_mod_accs[t].items():
                    self.log(
                        f'embed_val_{t}_{comp_name}_mask_accuracy',
                        acc,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                    # Log lift (multiple over proportional chance baseline)
                    lift = self._compute_lift(t, comp_name, acc)
                    self.log(
                        f'embed_val_{t}_{comp_name}_mask_lift',
                        lift,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
    
        if mlm_losses:
            loss_recon = finite_reduce_loss(mlm_losses)
        else:
            loss_recon = torch.tensor(0.0, device=self.device, dtype=self.log_var_recon.dtype)

        # --- CLIP-MoCo cross-modal contrastive loss ---
        loss_clip, clip_metrics = self._clip_moco_loss(batch, return_metrics=True)

        lambda_recon = float(self.config['training'].get('recon_meta_w',
                            self.config['training'].get('dna_recon_meta_weight', 1.0)))
        lambda_clip = float(self.config['training'].get('clip_meta_w',
                            self.config['training'].get('w_clip', 0.5)))

        prec_recon = torch.exp(-self.log_var_recon)
        prec_clip = torch.exp(-self.log_var_clip)

        total = lambda_recon * (prec_recon * loss_recon + self.log_var_recon) \
              + lambda_clip * (prec_clip * loss_clip + self.log_var_clip)
    
        # build validation log dict
        val_log = {
            'val_loss': total.detach(),
            'val_recon_loss': loss_recon.detach(),
            'val_clip_loss': loss_clip.detach(),
        }
        # per-component mask accuracies
        for t, accs in per_mod_accs.items():
            for comp_name, acc in accs.items():
                val_log[f'val_mask_acc__{t}__{comp_name}'] = float(acc)
        for k, v in clip_metrics.items():
            # Only log numeric values (skip strings like 'clip_mode' and 'clip_same_mod')
            if isinstance(v, (int, float, torch.Tensor)):
                val_log[f'val_{k}'] = v
        if log:
            self.log_dict(val_log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return total
        else:
            return total, val_log


    @torch.no_grad()
    def _momentum_update_teacher(self, step: Optional[int] = None):
        total_steps = self._total_opt_steps or self._total_steps_fallback()
        if step is None:
            step = self._current_opt_step()

        g = min(max(step / max(total_steps, 1), 0.0), 1.0)
        m = self.clip_m_final + 0.5 * (self.clip_m_base - self.clip_m_final) * (1.0 + math.cos(math.pi * g))

        def ema(qm, km):
            for pq, pk in zip(qm.parameters(), km.parameters()):
                pk.data.mul_(m).add_(pq.data.detach(), alpha=1.0 - m)

        ema(self.streams, self.streams_t)
        if self.stitchers is not None:
            ema(self.stitchers, self.stitchers_t)
        # EMA update for CLIP-MoCo projection heads
        for mod in self.modalities:
            ema(self.clip_heads_q[mod], self.clip_heads_k[mod])


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._momentum_update_teacher(step=self._current_opt_step())
        with torch.no_grad():
            self.log_var_recon.clamp_(-2.0, 3.0)
            self.log_var_clip.clamp_(-2.0, 3.0)


    def configure_optimizers(self):
        optimizer_type = self.config['training'].get('optimizer', 'AdamW')
        learning_rate = float(self.config['training'].get('learning_rate', 5e-5))
        weight_decay = float(self.config['training'].get('weight_decay', 1e-4))
        training_cfg = self.config.setdefault('training', {})
        training_len = int(training_cfg.get('training_len', 1))
        if training_len <= 0:
            training_len = 1
        training_cfg['training_len'] = training_len

        # separate CLIP-MoCo student head group and apply weight-decay exclusions
        clip_params = []
        for mod in self.modalities:
            clip_params.extend(list(self.clip_heads_q[mod].parameters()))
        clip_param_ids = {id(p) for p in clip_params}
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad or id(param) in clip_param_ids:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        clip_mult = float(self.config['training'].get('clip_head_lr_mult', 3.0))
        clip_wd = float(self.config['training'].get('clip_head_decay', 5e-4))

        param_groups = []
        if decay_params:
            param_groups.append({'params': decay_params, 'lr': learning_rate, 'weight_decay': weight_decay})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0})
        if clip_params:
            param_groups.append({'params': clip_params, 'lr': learning_rate * clip_mult, 'weight_decay': clip_wd})

        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(param_groups, lr=learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_type}')

        self._optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        self._lr_scheduler_placeholder = scheduler

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }


###################################################################################

class TaskProjection(nn.Module):
    """Per-task projection to decouple gradients from shared embedding."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OncoformerPost(LitOncoformerBase):
    def __init__(self, backbone: nn.Module,
                 config: Dict[str, Any],
                 checkpoint_dir: str = 'checkpoints'):
        super().__init__(config, checkpoint_dir)
        self.parent_config = cp.deepcopy(self.config)
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

        metadata_cfg = self.config.get('metadata') or {}
        metadata_data_cfg = metadata_cfg.get('data') or {}
        metadata_architecture_cfg = metadata_cfg.get('architecture') or {}

        self.use_confounder = metadata_data_cfg.get('confounder', None) is not None
        if self.use_confounder:
            conf_dim = int(metadata_architecture_cfg.get('confounder_dim', 0))
            self.embed_dim += conf_dim
        self.config['embedding_model'] = {
            'type': type(self.backbone).__name__,
            'config': cp.deepcopy(self.backbone.config)
        }
        self.fine_tune_model = self.config['training'].get('fine_tune_model', False)
        self.pre_train_model = self.config['training'].get('pre_train_model', False)
       
        if not (self.fine_tune_model or self.pre_train_model):
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval() # Set to eval if frozen
        
        self.config_hash = self.get_config_hash()
        self.config_checkpoint_dir = os.path.join(self.checkpoint_dir, f'post_{self.config_hash}')
        os.makedirs(self.config_checkpoint_dir, exist_ok=True)
        config_file = os.path.join(self.config_checkpoint_dir, 'config.yaml')
        if not os.path.exists(config_file):
            OmegaConf.save(config=self.config, f=config_file)
        parent_config_file = os.path.join(self.config_checkpoint_dir, 'parent_config.yaml')
        if not os.path.exists(parent_config_file):
            OmegaConf.save(config=self.parent_config, f=parent_config_file)

        vocab_cfg = metadata_cfg.get('vocab', {})
        vocab_path = vocab_cfg.get('file', './datasets/vocab_metadata_posttraining.json')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Metadata vocab file not found: {vocab_path}")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
                
        self.prediction_heads = nn.ModuleDict()
        self.task_projections = nn.ModuleDict()
        self.log_vars = nn.ParameterDict()
        self.meta_weights = {} 
        self.ignore_ids = {}
        self._optimizer: Optional[optim.Optimizer] = None
        self._lr_scheduler_placeholder: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self._total_opt_steps: Optional[int] = None

        # Top-level class weighting controls
        cw_cfg = metadata_cfg.get('class_weighting', {})
        cw_enabled = bool(cw_cfg.get('enabled', False))
        cw_use_in_loss = bool(cw_cfg.get('use_in_loss', True))
        cw_apply_to = cw_cfg.get('apply_to', 'all')

        for name, task_config in metadata_data_cfg.get('discrete', {}).items():
            if name not in self.vocab or 'levels' not in self.vocab[name]:
                raise KeyError(f"Metadata vocab missing levels for task '{name}'")
            n_classes = len(self.vocab[name]['levels'])

            # Determine if this task should use class weights
            task_use = task_config.get('use_class_weights', cw_use_in_loss)
            apply_allowed = (cw_apply_to == 'all') or (isinstance(cw_apply_to, (list, tuple)) and name in cw_apply_to)
            use_weights = cw_enabled and task_use and apply_allowed

            class_weights_tensor = None
            if use_weights:
                cw_list = task_config.get('class_weights', None)
                if isinstance(cw_list, (list, tuple)) and len(cw_list) > 0:
                    w = torch.tensor(cw_list, dtype=torch.float)
                    # Ensure correct length
                    if w.numel() < n_classes:
                        pad = torch.ones(n_classes - w.numel(), dtype=w.dtype)
                        w = torch.cat([w, pad], dim=0)
                    elif w.numel() > n_classes:
                        w = w[:n_classes]
                    class_weights_tensor = w

            # Per-task projection to decouple gradients from shared CLS embedding
            proj_dim = task_config.get('projection_dim', self.embed_dim)
            self.task_projections[name] = TaskProjection(self.embed_dim, proj_dim)
            self.prediction_heads[name] = Classification(proj_dim, n_classes, class_weights=class_weights_tensor)
            self.log_vars[name] = nn.Parameter(torch.tensor(0.0))
            self.meta_weights[name] = task_config.get('meta_weight', 1.0)
            ignore_patterns = task_config.get('ignore_patterns', [])
            if ignore_patterns and 'levels' in self.vocab[name]:
                ids_to_ignore = []
                for label_id, label_name in enumerate(self.vocab[name]['levels']):
                    if any(pattern in label_name.lower() for pattern in ignore_patterns):
                        ids_to_ignore.append(label_id)
                self.ignore_ids[name] = ids_to_ignore
            
        for name, task_config in metadata_data_cfg.get('point_estimate', {}).items():
            if name not in self.vocab or 'features' not in self.vocab[name]:
                raise KeyError(f"Metadata vocab missing features for task '{name}'")
            n_genes = len(self.vocab[name]['features'])
            # Per-task projection to decouple gradients from shared CLS embedding
            proj_dim = task_config.get('projection_dim', self.embed_dim)
            self.task_projections[name] = TaskProjection(self.embed_dim, proj_dim)
            self.prediction_heads[name] = Regression(proj_dim, n_genes)
            self.log_vars[name] = nn.Parameter(torch.tensor(0.0))
            self.meta_weights[name] = task_config.get('meta_weight', 1.0)
            
        for name, task_config in metadata_data_cfg.get('series', {}).items():
            if name not in self.vocab:
                raise KeyError(f"Metadata vocab missing series task '{name}'")
    
        
    def _get_task_type(self, name: str) -> str:
        data_config = self.config.get('metadata', {}).get('data', {})
        if name in data_config.get('series', []):
            return 'series'
        elif name in data_config.get('point_estimate', []):
            return 'point_estimate'
        elif name in data_config.get('discrete', []):
            return 'discrete'
        else:
            raise ValueError(f'Unknown task type: {name}')

    def forward(self, batch: Dict[str, Any]):
        if (self.fine_tune_model or self.pre_train_model):
            tumor_embeddings = self.backbone.calculate_pooled_embedding(batch)
        else:
            self.backbone.eval()
            with torch.no_grad():
                tumor_embeddings = self.backbone.calculate_pooled_embedding(batch)
        
        if self.use_confounder:
            tumor_embeddings = torch.cat([tumor_embeddings, batch['sample_metadata']['confounder']], dim=1)
        
        predictions_dict = {}
        for name, head_module in self.prediction_heads.items():
            task_type = self._get_task_type(name)
            # Apply per-task projection to decouple gradients from shared embedding
            proj_embedding = self.task_projections[name](tumor_embeddings)
            if task_type != 'series':
                predictions = head_module(proj_embedding)
            else:
                doses = batch['sample_metadata'][(task_type, name, 'doses')]
                predictions = head_module.forward_pass(proj_embedding, doses)
            predictions_dict[name] = predictions
        
        return predictions_dict
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        results_dict = self(batch)
        predictions_dict = {}
        predictions_dict['sample_metadata'] = batch['sample_metadata']['sample_metadata']
        for name, predictions in results_dict.items():
            task_type = self._get_task_type(name)
            predictions_dict[(name, 'predictions')] = predictions
            predictions_dict[(name, 'targets')] = batch['sample_metadata'][(task_type, name, 'targets')]
            if task_type == 'series':
                predictions_dict[(name, 'doses')] = batch['sample_metadata'][(task_type, name, 'doses')]
        return predictions_dict

    def _common_step(self, batch: Dict[str, Any], batch_idx: int):
        if (self.fine_tune_model or self.pre_train_model):
            tumor_embeddings = self.backbone.calculate_pooled_embedding(batch)
        else:
            self.backbone.eval()
            with torch.no_grad():
                tumor_embeddings = self.backbone.calculate_pooled_embedding(batch)

        if self.use_confounder:
            tumor_embeddings = torch.cat([tumor_embeddings, batch['sample_metadata']['confounder']], dim=1)

        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict_log = {}
        for name, head_module in self.prediction_heads.items():
            task_type = self._get_task_type(name)
            targets = batch['sample_metadata'][(task_type, name, 'targets')]
            
            ids_to_ignore = self.ignore_ids.get(name, [])
            if ids_to_ignore:
                for ignored_id in ids_to_ignore:
                    targets[targets == ignored_id] = -100
            
            if task_type == 'discrete':
                targets = targets.squeeze(-1).long()
            else:
                targets = targets.float()
            
            # Apply per-task projection to decouple gradients from shared embedding
            proj_embedding = self.task_projections[name](tumor_embeddings)
            
            if task_type == 'series':
                doses = batch['sample_metadata'][(task_type, name, 'doses')]
                batch_data = (proj_embedding, doses, targets)
            else:
                batch_data = (proj_embedding, targets)
            loss = head_module._common_step(batch=batch_data, batch_idx=batch_idx)

            precision = torch.exp(-self.log_vars[name])
            weighted_loss = precision * loss + self.log_vars[name]
            meta_weight = self.meta_weights.get(name, 1.0)

            loss_dict_log[f'{name}_loss'] = loss.detach()
            total_loss += weighted_loss * meta_weight
        
        return total_loss, loss_dict_log

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if self.trainer is None:
            self._total_opt_steps = None
            if self.pre_train_model and hasattr(self.backbone, "_total_opt_steps"):
                fallback = self.backbone._total_steps_fallback() if hasattr(self.backbone, "_total_steps_fallback") else None
                self.backbone._total_opt_steps = fallback
            if self.pre_train_model and hasattr(self.backbone, "_center_buf"):
                self.backbone._center_buf = []
            return

        total_opt_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        self._total_opt_steps = max(total_opt_steps, 1)

        training_cfg = self.config.setdefault('training', {})
        warmup_steps_cfg = training_cfg.get('scheduler_warmup_steps')
        if warmup_steps_cfg is None:
            warmup_fraction = float(training_cfg.get('scheduler_warmup_fraction', 0.1))
            warmup_steps = int(self._total_opt_steps * warmup_fraction)
            training_cfg['scheduler_warmup_steps'] = warmup_steps
        else:
            warmup_steps = int(warmup_steps_cfg)

        scheduler_type = training_cfg.get('scheduler_curve', 'cosine')
        optimizer = self._optimizer
        if optimizer is None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            self._optimizer = optimizer
        if optimizer is None:
            return

        if scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self._total_opt_steps,
            )
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self._total_opt_steps,
            )
        else:
            raise ValueError(f'Unsupported scheduler type: {scheduler_type}')

        self._scheduler = scheduler
        configs = getattr(self.trainer.strategy, "_lr_scheduler_configs", None)
        if configs:
            configs[0].scheduler = scheduler
            configs[0].interval = "step"
            configs[0].frequency = 1
        self._lr_scheduler_placeholder = scheduler

        if self.pre_train_model:
            if hasattr(self.backbone, "_total_opt_steps"):
                self.backbone._total_opt_steps = self._total_opt_steps
            if hasattr(self.backbone, "_center_buf"):
                self.backbone._center_buf = []

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        #print(f"{batch_idx} inside OncoformerPost training_step")
        pretrain_log = {}
        if self.pre_train_model:
            if hasattr(self.backbone, "_external_step"):
                self.backbone._external_step = int(self.global_step)
            pretrain_out = self.backbone.training_step(batch, batch_idx, log=False)
            if isinstance(pretrain_out, tuple):
                pretrain_loss, pretrain_log = pretrain_out
            else:
                pretrain_loss, pretrain_log = pretrain_out, {}
        loss, loss_dict_log = self._common_step(batch, batch_idx)
        if self.pre_train_model:
            loss += pretrain_loss
            loss_dict_log.update(pretrain_log)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, val in loss_dict_log.items():
            if name.startswith('train_'):
                name = f'embed_{name}'
            else:
                name = f'train_{name}'
            self.log(name, val, prog_bar=True, on_step=True, on_epoch=True)
        accumulate = 1
        if self.trainer is not None:
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        return loss / accumulate if accumulate > 1 else loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        pretrain_log = {}
        if self.pre_train_model:
            pretrain_out = self.backbone.validation_step(batch, batch_idx, log=False)
            if isinstance(pretrain_out, tuple):
                pretrain_loss, pretrain_log = pretrain_out
            else:
                pretrain_loss, pretrain_log = pretrain_out, {}
        loss, loss_dict_log = self._common_step(batch, batch_idx)
        if self.pre_train_model:
            loss += pretrain_loss
            loss_dict_log.update(pretrain_log)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, val in loss_dict_log.items():
            if name.startswith('val_'):
                name = f'embed_{name}'
            else:
                name = f'val_{name}'
            self.log(name, val, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, loss_dict_log = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        for name, val in loss_dict_log.items():
            self.log(f'test_{name}', val, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer_type = self.config['training'].get('optimizer', 'AdamW')
        learning_rate = self.config['training'].get('learning_rate', 0.003)
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        training_cfg = self.config.setdefault('training', {})
        training_len = int(training_cfg.get('training_len', 1))
        if training_len <= 0:
            training_len = 1
        training_cfg['training_len'] = training_len

        def split_decay(named_params):
            decay, no_decay = [], []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        param_groups: List[Dict[str, Any]] = []

        if self.pre_train_model:
            decay, no_decay = split_decay(list(self.named_parameters()))
            if decay:
                param_groups.append({'params': decay, 'lr': learning_rate, 'weight_decay': weight_decay})
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': learning_rate, 'weight_decay': 0.0})
        else:
            decay, no_decay = split_decay(list(self.prediction_heads.named_parameters()))
            if decay:
                param_groups.append({'params': decay, 'lr': learning_rate, 'weight_decay': weight_decay})
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': learning_rate, 'weight_decay': 0.0})

            if self.fine_tune_model:
                fine_tune_learning_rate = self.config['training'].get('fine_tune_model_learning_rate', 1e-6)
                decay, no_decay = split_decay(list(self.backbone.named_parameters()))
                if decay:
                    param_groups.append({'params': decay, 'lr': fine_tune_learning_rate, 'weight_decay': weight_decay})
                if no_decay:
                    param_groups.append({'params': no_decay, 'lr': fine_tune_learning_rate, 'weight_decay': 0.0})

        # Detect if model has CLIP-MoCo heads, and then freeze teacher and adjust LR accordingly
        if (self.pre_train_model or self.fine_tune_model) and hasattr(self.backbone, 'clip_heads_q'):
            # Freeze teacher heads
            if hasattr(self.backbone, 'clip_heads_k'):
                for mod in self.backbone.modalities:
                    for p in self.backbone.clip_heads_k[mod].parameters():
                        p.requires_grad = False

            clip_params = []
            for mod in self.backbone.modalities:
                clip_params.extend(list(self.backbone.clip_heads_q[mod].parameters()))
            clip_ids = {id(p) for p in clip_params}

            # Strip student params out of existing groups
            new_groups = []
            for g in param_groups:
                prms = [p for p in g['params'] if id(p) not in clip_ids]
                if prms:  # keep only non-empty groups
                    g = dict(g)
                    g['params'] = prms
                    new_groups.append(g)
            param_groups = new_groups

            # Scale from the right base LR
            base_lr_for_model = (
                self.config['training'].get('fine_tune_model_learning_rate', learning_rate)
                if (self.fine_tune_model and not self.pre_train_model)
                else learning_rate
            )
            clip_mult = float(self.config['training'].get('clip_head_lr_mult', 3.0))
            clip_wd = float(self.config['training'].get('clip_head_decay', 1e-4))
            param_groups.append({'params': clip_params,
                                 'lr': base_lr_for_model * clip_mult,
                                 'weight_decay': clip_wd})      
        
        # Handle stitcher-specific LR multiplier for fusion training
        if (self.pre_train_model or self.fine_tune_model) and hasattr(self.backbone, 'stitchers'):
            if self.backbone.stitchers is not None and len(self.backbone.stitchers) > 0:
                stitcher_params = [p for s in self.backbone.stitchers for p in s.parameters() if p.requires_grad]
                if stitcher_params:
                    stitcher_ids = {id(p) for p in stitcher_params}
                    
                    # Strip stitcher params from existing groups
                    new_groups = []
                    for g in param_groups:
                        prms = [p for p in g['params'] if id(p) not in stitcher_ids]
                        if prms:
                            g = dict(g)
                            g['params'] = prms
                            new_groups.append(g)
                    param_groups = new_groups
                    
                    # Add stitcher group with multiplied LR
                    stitch_mult = float(self.config['training'].get('stitch_lr_mult', 1.0))
                    param_groups.append({
                        'params': stitcher_params,
                        'lr': learning_rate * stitch_mult,
                        'weight_decay': weight_decay
                    })
        
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(param_groups, lr=learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

        self._optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        self._lr_scheduler_placeholder = scheduler

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        backbone = getattr(self, "backbone", None)
        if backbone is not None and hasattr(backbone, "_momentum_update_teacher"):
            backbone._momentum_update_teacher(step=int(self.global_step))

        with torch.no_grad():
            for name in self.log_vars.keys():
                self.log_vars[name].clamp_(-2.0, 3.0)
    
    def on_train_epoch_start(self):
        """Handle staged stream freezing for fusion training from transferred weights."""
        freeze_epochs = int(self.config['training'].get('freeze_streams_epochs', 0))
        if freeze_epochs <= 0:
            return
            
        backbone = getattr(self, 'backbone', None)
        if backbone is None or not hasattr(backbone, 'streams'):
            return
            
        modalities = getattr(backbone, 'modalities', [])
        if not modalities:
            return
            
        if self.current_epoch < freeze_epochs:
            # Freeze streams (but keep stitchers trainable)
            for m in modalities:
                if m in backbone.streams:
                    for p in backbone.streams[m].parameters():
                        p.requires_grad = False
        elif self.current_epoch == freeze_epochs:
            # Unfreeze streams at the designated epoch
            for m in modalities:
                if m in backbone.streams:
                    for p in backbone.streams[m].parameters():
                        p.requires_grad = True
            print(f"[Epoch {self.current_epoch}] Unfreezing streams for fine-tuning")

            
