import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np


class MixtureBatchSampler:
    """Mixture sampler that exhausts matched samples once per epoch.

    Cohorts are inferred dynamically based on modality presence:
      * ``matched`` – all configured modalities present
      * ``{mod}_unmatched`` – present in ``mod`` but missing at least one other modality

    During an epoch we iterate through every matched sample exactly once
    (without replacement). Unmatched cohorts are sampled with replacement to
    satisfy the requested proportion, but iteration stops as soon as matched
    samples are exhausted – even if many unmatched samples remain unseen.
    """

    def __init__(
        self,
        *,
        dataset,
        sampler_cfg: Dict,
        indices: Optional[Sequence[int]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.dataset = dataset
        self.cfg = sampler_cfg or {}
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.epoch_size = int(self.cfg.get("epoch_size", 0))  # 0 ⇒ auto
        self.shuffle_matched = bool(self.cfg.get("shuffle_matched", True))
        self.shuffle_unmatched_scope = bool(self.cfg.get("shuffle_unmatched_scope", True))
        self.stratify = list(self.cfg.get("stratify", []))
        self.rng = rng or random.Random()

        if indices is None:
            indices = list(range(len(dataset)))
        self.scope_indices = list(indices)

        # Fast lookup: sample_id -> dataset index order
        self.sample_ids = list(map(str, dataset.sample_metadata.index.astype(str).tolist()))
        self.sample_id_to_index = {sid: i for i, sid in enumerate(self.sample_ids)}

        modalities = list(dataset.modalities_list)
        present_by_mod: Dict[str, set] = {}
        for mod in modalities:
            tok = dataset.tokenized_by_mod.get(mod, {}) or {}
            present_by_mod[mod] = set(map(str, tok.keys()))

        index_set = set(self.scope_indices)
        self.cohort_pools: Dict[str, List[int]] = {}

        # matched: intersection across modalities
        matched_ids = set(self.sample_ids)
        for mod in modalities:
            matched_ids = matched_ids.intersection(present_by_mod[mod])
        matched_idx = [self.sample_id_to_index[s] for s in matched_ids if s in self.sample_id_to_index]
        matched_idx = [i for i in matched_idx if i in index_set]
        self.cohort_pools["matched"] = matched_idx

        # unmatched cohorts per modality
        for mod in modalities:
            unmatched_ids = present_by_mod[mod].difference(matched_ids)
            unmatched_idx = [self.sample_id_to_index[s] for s in unmatched_ids if s in self.sample_id_to_index]
            unmatched_idx = [i for i in unmatched_idx if i in index_set]
            self.cohort_pools[f"{mod}_unmatched"] = unmatched_idx

        self.pool_weights: Dict[str, float] = {}
        cohorts_cfg = self.cfg.get("cohorts", {}) or {}
        matched_weight = float(cohorts_cfg.get("matched", 0.0))
        if matched_idx:
            self.pool_weights["matched"] = matched_weight if matched_weight > 0 else 1.0

        unmatched_cfg = cohorts_cfg.get("unmatched", {}) or {}
        for mod, weight in unmatched_cfg.items():
            pool_name = f"{mod}_unmatched"
            if self.cohort_pools.get(pool_name):
                w = float(weight)
                if w > 0:
                    self.pool_weights[pool_name] = w

        if not self.pool_weights:
            # Fallback: uniform weights for any non-empty pool in scope
            for name, pool in self.cohort_pools.items():
                if pool:
                    self.pool_weights[name] = 1.0

        self._cached_plan: Optional[List[List[int]]] = None
        self._cached_plan_samples: int = 0

    # ------------------------------------------------------------------ helpers
    def _pool_for(self, cohort_name: str) -> List[int]:
        return self.cohort_pools.get(cohort_name, [])

    def _normalized_probs(self, cohort_names: List[str]) -> List[float]:
        weights = [max(self.pool_weights.get(name, 0.0), 0.0) for name in cohort_names]
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(cohort_names)] * len(cohort_names)
        return [w / total for w in weights]

    def _sample_unmatched(self, cohort_name: str) -> Optional[int]:
        base = self._pool_for(cohort_name)
        if not base:
            return None
        if not self.stratify:
            # Fast path: sampling with replacement, no stratification -> pick
            # directly from the pool. Avoids the O(len(pool)) list() copy that the
            # stratify path needs, which otherwise makes plan build O(n*budget).
            return self.rng.choice(base)
        pool = self._apply_stratify(list(base))
        if not pool:
            pool = list(base)
        return self.rng.choice(pool)

    def _pop_matched(self, matched_pool: List[int]) -> Optional[int]:
        if not matched_pool:
            return None
        if not self.stratify:
            # Fast path: no stratification, so every remaining element is equally
            # valid. Swap-remove a uniformly random index in O(1) instead of the
            # O(n) list() copy + O(n) list.remove() this used to do on every pop,
            # which made a single epoch-plan build O(n^2) (~10 min at n=318k for
            # DX1/DX2, and rebuilt every epoch). Distribution is identical: a
            # uniform draw without replacement.
            j = self.rng.randrange(len(matched_pool))
            idx = matched_pool[j]
            matched_pool[j] = matched_pool[-1]
            matched_pool.pop()
            return idx
        pool = self._apply_stratify(list(matched_pool))
        if not pool:
            pool = matched_pool
        idx = self.rng.choice(pool)
        matched_pool.remove(idx)
        return idx

    def _apply_stratify(self, pool: List[int]) -> List[int]:
        if not pool or not self.stratify:
            return pool
        meta = self.dataset.sample_metadata
        for rule in self.stratify:
            key = rule.get("key")
            if key is None or key not in meta.columns:
                continue
            scope = rule.get("scope", "global")
            target = rule.get("target_weights", {}) or {}
            default = rule.get("default", "proportional")

            if scope == "within_cohort":
                scope_idx = pool
            else:
                scope_idx = self.scope_indices

            groups = meta.iloc[scope_idx][key].astype(str)
            counts = groups.value_counts().to_dict()
            group_values = list(counts.keys())

            probs = []
            for gv in group_values:
                if gv in target:
                    probs.append(float(target[gv]))
                    continue
                if default == "uniform":
                    probs.append(1.0)
                else:  # proportional
                    probs.append(float(counts.get(gv, 0)))
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]

            gv = self.rng.choices(group_values, weights=probs, k=1)[0]
            keep_mask = (meta.iloc[pool][key].astype(str).values == gv)
            filtered = [idx for idx, keep in zip(pool, keep_mask) if keep]
            if filtered:
                pool = filtered
        return pool

    def _trim_plan_to_samples(self, plan: List[List[int]], limit: int) -> List[List[int]]:
        if limit <= 0:
            return plan
        trimmed: List[List[int]] = []
        used = 0
        for batch in plan:
            if used >= limit:
                break
            remaining = limit - used
            if len(batch) <= remaining:
                trimmed.append(batch)
                used += len(batch)
            else:
                trimmed.append(batch[:remaining])
                used = limit
                break
        return [b for b in trimmed if b]

    def _build_plan_no_matched(self) -> List[List[int]]:
        if not self.scope_indices:
            return []
        scope = list(self.scope_indices)
        if self.shuffle_unmatched_scope:
            self.rng.shuffle(scope)
        batches = [scope[i : i + self.batch_size] for i in range(0, len(scope), self.batch_size)]
        return batches

    def _build_epoch_plan(self) -> List[List[int]]:
        matched_pool = list(self._pool_for("matched"))
        if self.shuffle_matched:
            self.rng.shuffle(matched_pool)

        matched_budget = len(matched_pool)
        matched_weight = self.pool_weights.get("matched", 0.0)

        unmatched_names = [name for name in self.pool_weights.keys() if name != "matched"]

        if matched_budget == 0:
            plan = self._build_plan_no_matched()
            if self.epoch_size:
                plan = self._trim_plan_to_samples(plan, self.epoch_size)
            return plan

        reference_weight = matched_weight if matched_weight > 0 else 1.0
        budgets: Dict[str, int] = {"matched": matched_budget}
        for name in unmatched_names:
            weight = max(self.pool_weights.get(name, 0.0), 0.0)
            if weight <= 0:
                continue
            target = int(math.ceil(matched_budget * (weight / reference_weight)))
            if target > 0:
                budgets[name] = target

        plan: List[List[int]] = []
        current: List[int] = []
        remaining = budgets.copy()
        matched_remaining = matched_budget
        matched_exhausted = matched_budget == 0

        while True:
            available = [
                name
                for name, rem in remaining.items()
                if rem > 0 and (name != "matched" or matched_pool)
            ]
            if not available:
                break

            probs = self._normalized_probs(available)
            chosen = self.rng.choices(available, weights=probs, k=1)[0]

            if chosen == "matched":
                idx = self._pop_matched(matched_pool)
                if idx is not None:
                    matched_remaining -= 1
                    if matched_remaining == 0:
                        matched_exhausted = True
                        remaining["matched"] = 0
            else:
                idx = self._sample_unmatched(chosen)

            if idx is None:
                remaining[chosen] = 0
                continue

            remaining[chosen] -= 1
            current.append(idx)

            if len(current) == self.batch_size:
                plan.append(current)
                current = []
                if matched_exhausted:
                    break

        if current:
            plan.append(current)

        if self.epoch_size:
            plan = self._trim_plan_to_samples(plan, self.epoch_size)

        return plan

    def _ensure_plan(self) -> List[List[int]]:
        if self._cached_plan is None:
            plan = self._build_epoch_plan()
            self._cached_plan = plan
            self._cached_plan_samples = sum(len(batch) for batch in plan)
        return self._cached_plan

    # ---------------------------------------------------------------- iteration
    def _rank_slice(self, plan: List[List[int]]) -> List[List[int]]:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                w = dist.get_world_size()
                # Drop the remainder so every rank yields an IDENTICAL number of
                # batches. plan[rank::w] hands the first (len%w) ranks one extra
                # batch; the short ranks then exhaust early and the others block
                # forever on the gradient allreduce (NCCL watchdog timeout).
                # Truncating to a multiple of w mirrors DistributedSampler(drop_last=True).
                n = (len(plan) // w) * w
                return plan[dist.get_rank():n:w]
        except Exception:
            pass
        return plan

    def __iter__(self):
        plan = self._rank_slice(self._ensure_plan())
        for batch in plan:
            yield list(batch)
        # Invalidate after a full pass to rebuild next epoch
        self._cached_plan = None
        self._cached_plan_samples = 0

    def __len__(self):
        plan = self._ensure_plan()
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                # floor, matching _rank_slice's drop-remainder behaviour so the
                # reported step count equals what every rank actually yields.
                return len(plan) // dist.get_world_size()
        except Exception:
            pass
        return len(plan)

    def epoch_sample_count(self) -> int:
        self._ensure_plan()
        return self._cached_plan_samples


