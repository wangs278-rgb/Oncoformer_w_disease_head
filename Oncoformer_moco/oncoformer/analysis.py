from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _to_list(x: Any) -> Optional[List[Any]]:
    """
    Convert a tensor/array/series-like object to a Python list if possible.
    Return None if conversion is not possible or x is None.
    """
    if x is None:
        return None
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return list(x)
    return None


"""
Top‑1 extractor has been removed in favor of top‑k only.
Use extract_topk_attn_tokens_and_scores instead.
"""


def derive_aa_change(
    tokens_df: pd.DataFrame,
    ref_col: str = "aa_ref",
    mut_col: str = "aa_mut",
) -> pd.Series:
    """
    Build an amino-acid change label like "A>V" from ref and mut columns.
    Returns a Series with None where either ref or mut is missing.
    """
    ref = tokens_df.get(ref_col)
    mut = tokens_df.get(mut_col)
    if ref is None or mut is None:
        return pd.Series([None] * len(tokens_df))
    result = (ref.astype("string") + ">" + mut.astype("string"))
    mask_na = ref.isna() | mut.isna()
    result.loc[mask_na] = None
    return result


# ---------------------------------------------------------------------------- #
# Top‑K extraction
# ---------------------------------------------------------------------------- #

def extract_topk_attn_tokens_and_scores(
    predictions: Sequence[Dict[str, Any]],
    components: Sequence[str],
    tokenizer: Any,
    modality: str = "dna",
    K: int = 3,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.Series]]:
    """
    Extract per‑sample top‑k attention tokens and scores from model predictions.

    Returns two dicts keyed by k=1..K:
      - tokens_by_k[k]: DataFrame with one column per component (str tokens or None)
      - scores_by_k[k]: Series with the per‑sample attention score for rank k

    Falls back to top‑1 for k=1 when top‑k payload is missing.
    """
    K = max(1, int(K))
    tokens_lists_by_k: Dict[int, Dict[str, List[pd.Series]]] = {
        k: {c: [] for c in components} for k in range(1, K + 1)
    }
    scores_lists_by_k: Dict[int, List[pd.Series]] = {k: [] for k in range(1, K + 1)}

    comp_i2t: Dict[str, Dict[int, str]] = getattr(tokenizer, "metadata_idx2token", {}) or {}

    def _ensure_tokens(entry: Dict[str, Any], comp: str):
        # Accept either pre‑decoded tokens, integer ids, or raw values
        toks = entry.get("tokens", None)
        if toks is not None:
            return toks
        ids = entry.get("ids", None)
        if ids is not None:
            i2t = comp_i2t.get(comp, {})
            # Convert tensors to Python lists
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            # ids can be a list or list‑of‑lists; map recursively one level
            if isinstance(ids, list) and (len(ids) > 0) and isinstance(ids[0], list):
                return [[i2t.get(int(x), "<unk>") for x in row] for row in ids]
            elif isinstance(ids, list):
                return [i2t.get(int(x), "<unk>") for x in ids]
            else:
                # scalar id
                return [i2t.get(int(ids), "<unk>")]
        # values → keep numeric where possible
        vals = entry.get("values", None)
        if vals is None:
            return None
        if hasattr(vals, "tolist"):
            vals = vals.tolist()
        return vals

    for p in predictions:
        n = 0
        meta = p.get("sample_metadata")
        if isinstance(meta, (pd.DataFrame, pd.Series)):
            n = len(meta)
        elif meta is not None:
            try:
                n = len(meta)
            except Exception:
                n = 0

        expl = (
            p.get("attn_explanations", {})
             .get("self", {})
             .get(modality, None)
        )

        if expl is None:
            for k in range(1, K + 1):
                for c in components:
                    tokens_lists_by_k[k][c].append(pd.Series([None] * n))
                scores_lists_by_k[k].append(pd.Series([None] * n))
            continue

        topk_payload = expl.get("topk", None)

        # Scores fallback for k=1 when topk missing
        score_top1 = _to_list(expl.get("score", None))
        if score_top1 is None:
            score_top1_series = pd.Series([None] * n)
        else:
            score_top1_series = pd.Series(list(score_top1))

        if topk_payload is None:
            # Only top‑1 available in legacy payload
            scores_lists_by_k[1].append(score_top1_series)
            per_comp = expl.get("components", {})
            for c in components:
                entry = per_comp.get(c, {})
                toks = entry.get("tokens", None)
                if toks is None:
                    ids = _to_list(entry.get("ids", None))
                    if ids is None:
                        tokens_lists_by_k[1][c].append(pd.Series([None] * n))
                        continue
                    i2t = comp_i2t.get(c, {})
                    toks = [i2t.get(int(x), "<unk>") for x in ids]
                tokens_lists_by_k[1][c].append(pd.Series(list(toks)))
            for k in range(2, K + 1):
                for c in components:
                    tokens_lists_by_k[k][c].append(pd.Series([None] * n))
                scores_lists_by_k[k].append(pd.Series([None] * n))
            continue

        # We have full top‑k payload
        # scores: shape [B, K]
        scores = topk_payload.get("scores", None)
        if scores is not None and hasattr(scores, "tolist"):
            scores = scores.tolist()
        # components: dict[comp] → {tokens|ids|values}: list‑of‑lists shape [B, K]
        per_comp_k = topk_payload.get("components", {})

        for k in range(1, K + 1):
            # Scores
            if scores is None or len(scores) == 0:
                scores_lists_by_k[k].append(pd.Series([None] * n))
            else:
                # take k‑th column
                kth = [row[k - 1] if (isinstance(row, list) and len(row) >= k) else None for row in scores]
                scores_lists_by_k[k].append(pd.Series(kth))

            # Tokens per component
            for c in components:
                entry = per_comp_k.get(c, {})
                toks_2d = _ensure_tokens(entry, c)
                if toks_2d is None or len(toks_2d) == 0:
                    tokens_lists_by_k[k][c].append(pd.Series([None] * n))
                    continue
                kth_tokens = []
                for row in toks_2d:
                    if isinstance(row, list) and len(row) >= k:
                        kth_tokens.append(row[k - 1])
                    else:
                        kth_tokens.append(None)
                tokens_lists_by_k[k][c].append(pd.Series(kth_tokens))

        # If K >= 1, also ensure k=1 aligned with legacy score when present
        if K >= 1 and score_top1_series is not None:
            # Favor explicit top‑k score but fill gaps from legacy score
            last = scores_lists_by_k[1][-1]
            if last.isna().any():
                scores_lists_by_k[1][-1] = last.fillna(score_top1_series)

    tokens_by_k: Dict[int, pd.DataFrame] = {
        k: pd.DataFrame({
            c: pd.concat(tokens_lists, ignore_index=True) if len(tokens_lists) > 0 else pd.Series(dtype="object")
            for c, tokens_lists in comp_dict.items()
        })
        for k, comp_dict in tokens_lists_by_k.items()
    }
    scores_by_k: Dict[int, pd.Series] = {
        k: pd.concat(series_list, ignore_index=True) if len(series_list) > 0 else pd.Series(dtype="float")
        for k, series_list in scores_lists_by_k.items()
    }
    return tokens_by_k, scores_by_k

