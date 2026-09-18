"""
Code lifted verbatim (with minor import/path adjustments) from
https://github.com/czi-ai/GREmLN at commit e1c5d8e (MIT licensed).

The original project provides graph diffusion kernels and regulatory
network utilities for GREmLN. Here we vendor the relevant pieces so we
can reuse them inside Oncoformer without introducing their full stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Graph diffusion kernels (Chebyshev approximation)
# ---------------------------------------------------------------------------

def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _exp_kernel(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.exp(-beta * (x + 1))


def _cosine_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(torch.pi * x / 2)


def _remove_self_loops(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    row, col = edge_index
    mask = row != col
    edge_index = edge_index[:, mask]
    if edge_weight is None:
        return edge_index, None
    return edge_index, edge_weight[mask]


def dedupe_edges(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    *,
    directed: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Coalesce duplicate edges by summing their weights.

    Parameters
    ----------
    edge_index:
        Tensor with shape (2, E) containing source and destination indices.
    edge_weight:
        Optional tensor with length E. When None, the function treats every edge
        as weight 1.0 and returns `None` after deduplication.
    num_nodes:
        Number of nodes in the graph (used to size the COO tensor).
    directed:
        When False, each edge is canonicalized via (min(u, v), max(u, v)) before
        deduplication, effectively treating the graph as undirected.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_weight

    if not directed:
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        edge_index = torch.stack((u, v), dim=0)

    drop_weight = edge_weight is None
    if drop_weight:
        weights = torch.ones(
            edge_index.size(1),
            dtype=torch.float32,
            device=edge_index.device,
        )
    else:
        weights = edge_weight
        if weights.dtype != torch.float32:
            weights = weights.to(dtype=torch.float32)

    sparse = torch.sparse_coo_tensor(
        edge_index,
        weights,
        (num_nodes, num_nodes),
    ).coalesce()

    coalesced_index = sparse.indices()
    coalesced_weight = sparse.values()

    if drop_weight:
        return coalesced_index, None
    return coalesced_index, coalesced_weight


def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _rescaled_L(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    edge_index, edge_weight = _remove_self_loops(edge_index, edge_weight)

    if edge_index.shape[-1] == 0:
        idx = torch.arange(num_nodes, device=edge_index.device)
        edge_index = idx.unsqueeze(0).repeat(2, 1)

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1),
            dtype=torch.float32,
            device=edge_index.device,
        )
    row, col = edge_index[0], edge_index[1]

    deg = _scatter_sum(edge_weight, row, num_nodes).clamp(min=1e-8)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
    deg_inv_sqrt.masked_fill_(torch.isnan(deg_inv_sqrt), 0.0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    L_rescaled = torch.sparse_coo_tensor(
        edge_index,
        -edge_weight,
        (num_nodes, num_nodes),
    )
    return L_rescaled


def _chebyshev_coeff(
    L_rescaled: torch.Tensor,
    K: int,
    func,
    N: int = 100,
) -> torch.Tensor:
    ind = torch.arange(0, K + 1, dtype=torch.float32, device=L_rescaled.device)
    ratio = torch.pi * (torch.arange(1, N + 1, dtype=torch.float32, device=L_rescaled.device) - 0.5) / N
    x = torch.cos(ratio)
    T_kx = torch.cos(ind.view(-1, 1) * ratio)
    w = torch.ones(N, device=L_rescaled.device) * (torch.pi / N)
    f_x = func(x)
    c_k = (2 / torch.pi) * torch.matmul(T_kx, w * f_x)
    return c_k


@torch.amp.autocast(enabled=False, device_type="cuda")
def _chebyshev_diffusion_per_sample(
    edge_index: torch.Tensor,
    num_nodes: int,
    E: torch.Tensor,
    k: int = 128,
    edge_weight: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    kernel=_exp_kernel,
) -> torch.Tensor:
    L_rescaled = _rescaled_L(edge_index, num_nodes, edge_weight)
    c_k = _chebyshev_coeff(L_rescaled, k, lambda x: kernel(x, beta))
    E = E.to(torch.float32)
    s, h, d = E.size()
    if s != num_nodes:
        raise ValueError(f"Expect {num_nodes} nodes, got {s}")
    E_reshaped = E.reshape(num_nodes, h * d)
    c_k = c_k.to(torch.float32)
    T_0 = E_reshaped
    T_1 = torch.sparse.mm(L_rescaled, E_reshaped)
    y = c_k[0] * T_0 + c_k[1] * T_1

    T_k_prev = T_1
    T_k_prev_prev = T_0
    for idx in range(2, k + 1):
        T_k = 2 * torch.sparse.mm(L_rescaled, T_k_prev) - T_k_prev_prev
        y = y + c_k[idx] * T_k
        T_k_prev_prev = T_k_prev
        T_k_prev = T_k

    final_emb = y.reshape(num_nodes, h, d)
    return final_emb.to(E.dtype)


def _chebyshev_diffusion(
    edge_index_list: Sequence[torch.Tensor],
    num_nodes_list: Sequence[int],
    E: torch.Tensor,
    k: int = 64,
    beta: float = 0.5,
    edge_weight_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    kernel=_exp_kernel,
) -> torch.Tensor:
    B, S, H, D = E.size()
    results: List[torch.Tensor] = []

    for i in range(B):
        E_i = E[i, : num_nodes_list[i], ...]
        edge_index = edge_index_list[i]
        edge_weight = None if edge_weight_list is None else edge_weight_list[i]
        sample_emb = _chebyshev_diffusion_per_sample(
            edge_index,
            num_nodes_list[i],
            E_i,
            k=k,
            edge_weight=edge_weight,
            beta=beta,
            kernel=kernel,
        )

        pad_size = S - sample_emb.size(0)
        if pad_size > 0:
            zero_pad_right = torch.zeros(
                pad_size,
                H,
                D,
                device=E.device,
                dtype=sample_emb.dtype,
            )
            sample_emb = torch.cat([sample_emb, zero_pad_right], dim=0)
        results.append(sample_emb)

    final = torch.stack(results, dim=0)
    if final.size() != E.size():
        raise RuntimeError(f"Expect {E.size()}, got {final.size()}")
    return final


# ---------------------------------------------------------------------------
# Regulatory network utilities (minimal)
# ---------------------------------------------------------------------------

REG_NAME = "regulator.values"
TAR_NAME = "target.values"
WT_NAME = "mi.values"
LIK_NAME = "log.p.values"


@dataclass
class RegulatoryNetwork:
    regulators: pd.Series
    targets: pd.Series
    weights: pd.Series
    likelihoods: pd.Series

    def __post_init__(self):
        data = pd.DataFrame(
            {
                REG_NAME: self.regulators,
                TAR_NAME: self.targets,
                WT_NAME: self.weights,
                LIK_NAME: self.likelihoods,
            }
        ).astype({WT_NAME: float, LIK_NAME: float})
        self._df = data
        self.genes = set(self._df[REG_NAME]) | set(self._df[TAR_NAME])

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @classmethod
    def from_csv(
        cls,
        path: str,
        reg_name: str = REG_NAME,
        tar_name: str = TAR_NAME,
        wt_name: str = WT_NAME,
        lik_name: str = LIK_NAME,
        **kwargs,
    ) -> "RegulatoryNetwork":
        df = pd.read_csv(path, **kwargs)
        return cls(df[reg_name], df[tar_name], df[wt_name], df[lik_name])

    def prune(
        self,
        limit_regulon: Optional[int] = None,
        limit_graph: Optional[int] = None,
        inplace: bool = False,
    ) -> "RegulatoryNetwork":
        df = self._df if inplace else self._df.copy()

        if limit_regulon is not None:
            df = (
                df.sort_values(by=[REG_NAME, WT_NAME], ascending=[True, False])
                .groupby(REG_NAME, group_keys=False)
                .head(limit_regulon)
            )

        if limit_graph is not None:
            df = df.nlargest(limit_graph, WT_NAME)

        if inplace:
            self._df = df
            self.genes = set(df[REG_NAME]) | set(df[TAR_NAME])
            return self

        return RegulatoryNetwork(
            df[REG_NAME],
            df[TAR_NAME],
            df[WT_NAME],
            df[LIK_NAME],
        )

    def make_undirected(
        self,
        drop_unpaired: bool = False,
        inplace: bool = False,
    ) -> "RegulatoryNetwork":
        df = self._df if inplace else self._df.copy()
        edge_set = set(zip(df[REG_NAME], df[TAR_NAME]))
        reverse_set = {(t, r) for r, t in edge_set}

        if drop_unpaired:
            bidirectional = edge_set & reverse_set
            mask = [(r, t) in bidirectional for r, t in zip(df[REG_NAME], df[TAR_NAME])]
            df = df[mask].reset_index(drop=True)
        else:
            existing = set(zip(df[REG_NAME], df[TAR_NAME]))
            reversed_edges = []
            for _, row in df.iterrows():
                src, tgt = row[REG_NAME], row[TAR_NAME]
                if (tgt, src) not in existing:
                    reversed_edges.append(
                        {
                            REG_NAME: tgt,
                            TAR_NAME: src,
                            WT_NAME: row[WT_NAME],
                            LIK_NAME: row[LIK_NAME],
                        }
                    )
            if reversed_edges:
                reversed_df = pd.DataFrame(reversed_edges)
                df = pd.concat([df, reversed_df], ignore_index=True)

        if inplace:
            self._df = df
            self.genes = set(df[REG_NAME]) | set(df[TAR_NAME])
            return self

        return RegulatoryNetwork(
            df[REG_NAME],
            df[TAR_NAME],
            df[WT_NAME],
            df[LIK_NAME],
        )

    def subset_edges(
        self,
        genes: Iterable[str],
        preserve_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        genes = list(genes)
        local_map = {gene: idx for idx, gene in enumerate(genes)}
        df = self._df[
            self._df[REG_NAME].isin(local_map)
            & self._df[TAR_NAME].isin(local_map)
        ]
        regulators = df[REG_NAME].map(local_map).to_numpy()
        targets = df[TAR_NAME].map(local_map).to_numpy()
        edge_index = torch.tensor(
            [regulators, targets],
            dtype=torch.long,
        )

        if preserve_weights and WT_NAME in df:
            edge_weight = torch.tensor(df[WT_NAME].to_numpy(), dtype=torch.float32)
        else:
            edge_weight = None
        return edge_index, edge_weight


    def project_targets(
        self,
        genes: Iterable[str],
        *,
        min_shared: int = 1,
        weight: str = "sum",
        topk_per_node: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Project regulators onto their targets to induce target-target edges.
        Only targets present in `genes` are considered. Returns undirected graph.
        """
        genes = list(genes)
        if not genes or min_shared <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        local_map = {gene: idx for idx, gene in enumerate(genes)}
        df = self._df[self._df[TAR_NAME].isin(local_map)]
        if df.empty:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        weight_mode = str(weight).lower()
        allowed_weights = {"sum", "count", "product", "jaccard", "cosine"}
        if weight_mode not in allowed_weights:
            weight_mode = "sum"

        # Track regulator degrees for jaccard/cosine if needed
        regulator_target_map = {}

        from collections import defaultdict

        edge_scores: Dict[Tuple[int, int], float] = defaultdict(float)
        target_counts: Dict[int, float] = defaultdict(float)

        grouped = df.groupby(REG_NAME, sort=False)
        for regulator, sub in grouped:
            targets = [local_map[t] for t in sub[TAR_NAME]]
            if len(targets) < 2:
                continue
            weights = sub[WT_NAME].astype(float).tolist() if WT_NAME in sub else [1.0] * len(targets)

            regulator_target_map[regulator] = {
                "targets": targets,
                "weights": weights,
            }

            if weight_mode == "jaccard" or weight_mode == "cosine":
                for idx, t in enumerate(targets):
                    target_counts[t] += 1.0 if weight_mode == "jaccard" else abs(weights[idx])

            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    u, v = targets[i], targets[j]
                    if u == v:
                        continue
                    w_u = abs(weights[i])
                    w_v = abs(weights[j])
                    key = (min(u, v), max(u, v))
                    if weight_mode == "sum":
                        edge_scores[key] += w_u + w_v
                    elif weight_mode == "count":
                        edge_scores[key] += 1.0
                    elif weight_mode == "product":
                        edge_scores[key] += w_u * w_v
                    else:
                        # For jaccard/cosine, accumulate raw counts; normalization later
                        edge_scores[key] += 1.0

        if not edge_scores:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        # Apply min_shared filter
        filtered_items = []
        for key, score in edge_scores.items():
            if weight_mode in ("jaccard", "cosine"):
                # Normalization: |N(u) ∩ N(v)| for count-based; adjust when counts stored differently
                u, v = key
                shared = score
                if shared < min_shared:
                    continue
                if weight_mode == "jaccard":
                    denom = target_counts[u] + target_counts[v] - shared
                    if denom <= 0:
                        norm = 0.0
                    else:
                        norm = shared / denom
                else:  # cosine
                    denom = (target_counts[u] * target_counts[v]) ** 0.5
                    if denom <= 0:
                        norm = 0.0
                    else:
                        norm = shared / denom
                filtered_items.append((key, norm))
            else:
                if score >= float(min_shared):
                    filtered_items.append((key, score))

        if not filtered_items:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        if topk_per_node is not None and topk_per_node > 0:
            adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            for (u, v), score in filtered_items:
                adjacency[u].append((v, score))
                adjacency[v].append((u, score))

            kept_edges: Set[Tuple[int, int]] = set()
            for node, neighbors in adjacency.items():
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for tgt, _score in neighbors[: topk_per_node]:
                    kept_edges.add((min(node, tgt), max(node, tgt)))
            filtered_items = [item for item in filtered_items if item[0] in kept_edges]

            if not filtered_items:
                return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        rows: List[int] = []
        cols: List[int] = []
        weights_list: List[float] = []
        for (u, v), score in filtered_items:
            rows.extend([u, v])
            cols.extend([v, u])
            weights_list.extend([score, score])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights_list, dtype=torch.float32)
        return edge_index, edge_weight


