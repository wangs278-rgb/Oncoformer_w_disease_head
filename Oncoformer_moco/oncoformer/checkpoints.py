"""Utilities for working with Oncoformer checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

import torch

logger = logging.getLogger(__name__)


def load_stream_backbone(
    model: torch.nn.Module,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    *,
    modality: str,
    include_teacher: bool = False,
    strict: bool = False,
    map_location: str = "cpu",
) -> None:
    """Load weights for a single modality stream from a checkpoint.

    Parameters
    ----------
    model:
        An :class:`OncoformerOmics` instance, an :class:`OncoformerPost`, or any object exposing
        ``streams`` (and optionally ``streams_t``). If a wrapper supplies a ``backbone`` attribute
        with those members, it will be used automatically.
    checkpoint:
        Path to a checkpoint file or a pre-loaded state dict mapping parameter names to tensors.
    modality:
        Name of the modality stream to load (e.g. ``"dna"`` or ``"rna"``).
    include_teacher:
        When ``True`` and the checkpoint contains matching ``streams_t`` entries, load them as well.
    strict:
        Forwarded to :meth:`torch.nn.Module.load_state_dict`. Defaults to ``False`` to avoid errors
        when the checkpoint only partially overlaps with the current architecture.
    map_location:
        Passed to :func:`torch.load` when ``checkpoint`` is a path. Defaults to ``"cpu"`` so that
        weights can be loaded on CPU-only machines.

    Notes
    -----
    * Keys are matched by prefix: ``streams.{modality}`` (and optionally ``streams_t.{modality}``).
    * Missing keys are logged as warnings so callers can double-check architecture changes.
    * Parameters not present in the checkpoint are left untouched.
    """

    target = model
    if not hasattr(target, "streams"):
        backbone = getattr(model, "backbone", None)
        if backbone is not None and hasattr(backbone, "streams"):
            target = backbone
        else:
            raise AttributeError("Model must expose a 'streams' ModuleDict or provide a backbone with one")

    modality = str(modality)
    stream_prefix = f"streams.{modality}."
    teacher_prefix = f"streams_t.{modality}."
    stream_variants = [
        stream_prefix,
        f"backbone.{stream_prefix}",
        f"model.backbone.{stream_prefix}",
    ]
    teacher_variants = [
        teacher_prefix,
        f"backbone.{teacher_prefix}",
        f"model.backbone.{teacher_prefix}",
    ]

    state_dict: Mapping[str, torch.Tensor]
    if isinstance(checkpoint, (str, Path)):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=map_location)
        state_dict = _extract_state_dict(payload)
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint  # assume already a state dict
    else:
        raise TypeError("checkpoint must be a path or a mapping of parameter tensors")

    model_state = target.state_dict()

    def _copy_matching(base_prefix: str, variants: list[str]) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        loaded: list[str] = []
        for key, tensor in state_dict.items():
            matched_suffix = None
            for variant in variants:
                if key.startswith(variant):
                    matched_suffix = key[len(variant):]
                    break
            if matched_suffix is None:
                continue
            dest_key = f"{base_prefix}{matched_suffix}"
            if dest_key not in model_state:
                missing.append(dest_key)
                continue
            if model_state[dest_key].shape != tensor.shape:
                logger.warning(
                  "%s ignored due to the mismatched shape", dest_key
                )
                missing.append(dest_key)
                continue
            model_state[dest_key].copy_(tensor)
            loaded.append(dest_key)
        return loaded, missing

    loaded_keys, missing_keys = _copy_matching(stream_prefix, stream_variants)

    teacher_loaded: list[str] = []
    teacher_missing: list[str] = []
    if include_teacher and hasattr(target, "streams_t"):
        teacher_loaded, teacher_missing = _copy_matching(teacher_prefix, teacher_variants)

    # push weights back to model
    target.load_state_dict(model_state, strict=strict)

    if not loaded_keys:
        logger.warning(
            "No parameters loaded for modality '%s'. Check checkpoint contents and naming.",
            modality,
        )
    else:
        logger.info(
            "Loaded %d parameters into streams.%s (strict=%s)",
            len(loaded_keys),
            modality,
            strict,
        )

    if include_teacher:
        if teacher_loaded:
            logger.info(
                "Loaded %d parameters into streams_t.%s", len(teacher_loaded), modality
            )
        elif hasattr(target, "streams_t"):
            logger.warning(
                "Requested teacher weights for '%s' but none were loaded.", modality
            )

    if missing_keys and strict:
        logger.warning(
            "Missing %d parameter(s) for streams.%s while strict=True: %s",
            len(missing_keys),
            modality,
            missing_keys,
        )
    if include_teacher and teacher_missing and strict:
        logger.warning(
            "Missing %d teacher parameter(s) for streams_t.%s while strict=True: %s",
            len(teacher_missing),
            modality,
            teacher_missing,
        )


def _extract_state_dict(payload: Mapping[str, torch.Tensor] | Mapping[str, Mapping]) -> Mapping[str, torch.Tensor]:
    """Return the state dict from a Lightning-style checkpoint payload."""

    if isinstance(payload, MutableMapping):
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]
    if isinstance(payload, Mapping):
        return payload  # assume raw state dict already
    raise TypeError("Unsupported checkpoint payload type; expected mapping with parameters")
