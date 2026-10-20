from __future__ import annotations

import os

# Fix for some torch.compile interactions with CUDA Graphs
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

import contextlib
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import get_dataloaders
from data.proteins import get_dima_loader
from data.uniref50 import get_evodiff_uniref50_loader
from diffusion.continuous.logit_postprocess import _model_logits_continuous
from diffusion.continuous.losses import (
    binary_score_interpolation_loss,
    token_score_interpolation_loss,
)
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler, HeunSampler
from diffusion.discrete.losses import dwdse_loss
from diffusion.discrete.processes import DiscreteForwardProcess
from diffusion.discrete.samplers import TweedieTauLeapingSampler
from models import create_model
from utils.callbacks import (
    Callback,
    EntropySchedulePlotCallback,
    ExternalPPLCallback,
    MauveCallback,
    OfflineEntropyProfileCallback,
    SigmaDataEstimator,
    VisualizationCallback,
    VLBBoundCallback,
)
from utils.ecc_secded import ecc_chunk_len, ecc_from_cfg
from utils.ema import EMA
from utils.optim import get_optimizer_and_scheduler
from utils.schedule_controller import EntropyScheduleController
from utils.slurm_runtime import DeadlineGuard
from utils.tb_manager import TBManager

# Optional Weights & Biases
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    wandb = None
    WANDB_AVAILABLE = False


# ----------------------------------------------------------------------
# Global numeric / matmul / attention settings
# ----------------------------------------------------------------------

# TF32 matmul on Ampere: good speedup with negligible impact for DL
# workloads.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    # Let PyTorch choose faster kernels for fp32 matmuls.
    torch.set_float32_matmul_precision("high")
except AttributeError:
    # Older PyTorch versions.
    pass

torch._dynamo.config.cache_size_limit = 64  # or 128 if you want


def _enable_flash_sdp():
    """
    Enable PyTorch's flash / mem-efficient SDPA (scaled dot-product
    attention) backends. This does NOT change model architecture or
    checkpoints.
    """
    if not torch.cuda.is_available():
        print(
            "[FlashSDP] CUDA not available; keeping default attention "
            "backends."
        )
        return

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)  # keep math as a fallback
        # Only print on main process to avoid log spam (checked later)
    except AttributeError:
        print("[FlashSDP] SDPA backend toggles not found; PyTorch too old?")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _atomic_torch_save(state, final_path) -> None:
    """
    Write a checkpoint to a temp sibling then atomically rename it into
    place.
    """
    final_path = Path(final_path)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def _mm_replay_task_weights(cfg) -> dict[str, float]:
    """Task weights for the sequence-only replay loader."""
    override = getattr(cfg.data, "replay_task_weights", None)
    if override is not None:
        return {str(k): float(v) for k, v in dict(override).items()}
    return {"sequence_marginal": 1.0}


def _parse_mixture_phases(spec):
    """
    Config-friendly mixture phases -> the (progress, {(corpus, task):
    w}) form.
    """
    phases = []
    for start, weights in spec:
        parsed = {}
        for key, value in dict(weights).items():
            if isinstance(key, tuple):
                parsed[key] = float(value)
                continue
            corpus, _, task = str(key).partition(":")
            if not task:
                raise ValueError(
                    f"mixture weight key {key!r} must be 'corpus:task'"
                )
            parsed[(corpus, task)] = float(value)
        phases.append((float(start), parsed))
    return phases


_MM_VAL_SIGMA_EDGES = (0.002, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 80.0)


def _ema_decay_for_step(
    start: float, end: float, ramp_steps: int, step: int
) -> float:
    """Linear EMA-decay ramp, clamped at both endpoints."""
    if ramp_steps <= 0:
        return float(end)
    frac = min(max(float(step) / float(ramp_steps), 0.0), 1.0)
    return float(start + frac * (end - start))


def _mm_sigma_bin_label(lo: float, hi: float) -> str:
    def _part(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return f"{_part(lo)}_{_part(hi)}"


def _mm_validation_accumulator(device, tasks) -> dict[str, torch.Tensor]:
    prefixes = [f"validation/modality/{m}" for m in ("seq", "struct")]
    prefixes += [
        f"validation/task/{task}/{modality}"
        for task in sorted(tasks)
        for modality in ("seq", "struct")
    ]
    prefixes += [
        f"validation/sigma/{modality}/{_mm_sigma_bin_label(lo, hi)}"
        for modality in ("seq", "struct")
        for lo, hi in zip(
            _MM_VAL_SIGMA_EDGES[:-1], _MM_VAL_SIGMA_EDGES[1:], strict=True
        )
    ]
    # [EDM-weighted error sum, unweighted MSE sum, correct-bit sum, bit
    # count]
    return {
        p: torch.zeros(4, device=device, dtype=torch.float64) for p in prefixes
    }


def _mm_accumulate_validation(
    stats: dict[str, torch.Tensor], diagnostics: dict[str, object], task_names
) -> None:
    device = next(iter(stats.values())).device
    task_names = list(task_names)

    def _add(prefix: str, modality: str, select: torch.Tensor) -> None:
        count = diagnostics[f"{modality}_count"].to(
            device=device, dtype=torch.float64
        )
        select = select.to(device=device, dtype=torch.bool) & (count > 0)
        if not bool(select.any()):
            return
        stats[prefix][0] += (
            diagnostics[f"{modality}_edm_sum"]
            .to(device, torch.float64)[select]
            .sum()
        )
        stats[prefix][1] += (
            diagnostics[f"{modality}_mse_sum"]
            .to(device, torch.float64)[select]
            .sum()
        )
        stats[prefix][2] += (
            diagnostics[f"{modality}_correct_sum"]
            .to(device, torch.float64)[select]
            .sum()
        )
        stats[prefix][3] += count[select].sum()

    batch_size = len(task_names)
    all_rows = torch.ones(batch_size, device=device, dtype=torch.bool)
    for modality in ("seq", "struct"):
        _add(f"validation/modality/{modality}", modality, all_rows)

    for task in sorted(set(task_names)):
        rows = torch.tensor(
            [name == task for name in task_names], device=device
        )
        for modality in ("seq", "struct"):
            prefix = f"validation/task/{task}/{modality}"
            if prefix in stats:
                _add(prefix, modality, rows)

    for modality in ("seq", "struct"):
        sigma = diagnostics[f"sigma_{modality}"].to(device=device)
        for index, (lo, hi) in enumerate(
            zip(_MM_VAL_SIGMA_EDGES[:-1], _MM_VAL_SIGMA_EDGES[1:], strict=True)
        ):
            rows = (sigma >= lo) & (
                sigma <= hi
                if index == len(_MM_VAL_SIGMA_EDGES) - 2
                else sigma < hi
            )
            prefix = (
                f"validation/sigma/{modality}/{_mm_sigma_bin_label(lo, hi)}"
            )
            _add(prefix, modality, rows)


def _mm_finalize_validation(
    stats: dict[str, torch.Tensor],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix, values in stats.items():
        count = float(values[3].item())
        if count <= 0.0:
            continue
        metrics[f"{prefix}/edm_loss"] = float(values[0].item() / count)
        metrics[f"{prefix}/mse"] = float(values[1].item() / count)
        metrics[f"{prefix}/bit_acc"] = float(values[2].item() / count)
        metrics[f"{prefix}/target_bits"] = count

    seq = metrics.get("validation/modality/seq/mse")
    struct = metrics.get("validation/modality/struct/mse")
    if seq is not None and struct is not None:
        metrics["validation/balanced_mse"] = 0.5 * (seq + struct)
    return metrics


def _maybe_set_seed(cfg):
    """
    Set Python/NumPy/Torch seeds for reproducibility if cfg.train.seed
    is provided.
    """
    seed = getattr(cfg.train, "seed", None)
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow turning determinism off for speed. Default = True to
    # preserve existing behaviour.
    deterministic = bool(getattr(cfg.train, "deterministic", True))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def reseed_stochastic_stream(cfg, rank: int) -> int:
    """Give each rank its own noise stream."""
    base = getattr(cfg.train, "seed", None)
    if base is None:
        return -1
    stochastic_seed = (int(base) * 1_000_003 + int(rank)) % (2**31 - 1)
    random.seed(stochastic_seed)
    np.random.seed(stochastic_seed)
    torch.manual_seed(stochastic_seed)
    torch.cuda.manual_seed_all(stochastic_seed)
    return stochastic_seed


def _human_bytes(n_params: int, dtype: torch.dtype = torch.float32) -> str:
    """Approximate parameter memory footprint given dtype."""
    bytes_per = {
        torch.float64: 8,
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.bool: 1,
    }.get(dtype, 4)
    total = n_params * bytes_per
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.1f}{unit}"
        total /= 1024.0
    return f"{total:.1f}PB"


def _count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def _fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def _cfg_to_dict(obj) -> Any:
    """
    Recursively convert ml_collections.ConfigDict / nested containers
    into plain JSON-serializable Python objects.

    Handles:
      - ConfigDict / objects with .to_dict()
      - dict
      - list / tuple
      - pathlib.Path
      - torch.device / torch.dtype
      - numpy scalars / arrays
      - torch tensors (best-effort: convert to Python scalars/lists)
    """
    # ConfigDict-like
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _cfg_to_dict(obj.to_dict())

    # Plain dict
    if isinstance(obj, dict):
        return {str(k): _cfg_to_dict(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_dict(v) for v in obj]

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # torch device / dtype
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # torch tensors
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # JSON-safe primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Fallback for simple objects with __dict__
    if hasattr(obj, "__dict__"):
        return {
            str(k): _cfg_to_dict(v)
            for k, v in obj.__dict__.items()
            if not str(k).startswith("_")
        }

    # Last resort: stringify
    return str(obj)


def _save_config_to_run_dir(cfg, run_dir: Path):
    """
    Save the effective config of this run into:
      - config.json        : machine-readable fully materialized config
      - original_config.py : copy of the Python config used (first run)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _cfg_to_dict(cfg)

    cfg_json_path = run_dir / "config.json"
    with open(cfg_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)

    config_path = getattr(cfg, "_config_path", None)
    if config_path is not None and os.path.isfile(config_path):
        dst = run_dir / "original_config.py"
        if not dst.exists():
            shutil.copy2(config_path, dst)


class _NullWriter:
    """Drop-in TB writer that does nothing (safe for non-master ranks)."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def _unwrap_all(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and torch.compile wrappers until reaching the real module."""
    m = model
    while True:
        changed = False
        if hasattr(m, "module"):  # DDP wrapper usually outermost
            m = m.module
            changed = True
        if hasattr(m, "_orig_mod"):  # torch.compile wrapper
            m = m._orig_mod
            changed = True
        if not changed:
            break
    return m


def _gpu_mem_msg(device):
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.synchronize(device)
    a = torch.cuda.memory_allocated(device) / (1024**3)
    r = torch.cuda.memory_reserved(device) / (1024**3)
    return f"alloc={a:.2f}GiB reserved={r:.2f}GiB"


# -----------------------------------------------------------------------------
# Conditioning helpers (shared with training)
# -----------------------------------------------------------------------------
def _bits_per_unit(cfg) -> int:
    data = getattr(cfg, "data", object())

    ecc_cfg = getattr(data, "ecc", None)
    if ecc_cfg is not None and bool(getattr(ecc_cfg, "enabled", False)):
        ecc = ecc_from_cfg(cfg)
        return int(ecc_chunk_len(ecc))  # e.g. 21

    bpt = getattr(data, "bits_per_token", None)
    if bpt is not None:
        return int(bpt)
    return int(getattr(data, "bits_per_char", 1))


def _cond_len_bits_fixed(cfg, seq_len_bits: int) -> int:
    """
    Fixed prefix length in bits (backward compatible). Prefers
    cfg.cond.cond_len_tokens (semantic/BPE), else
    cfg.cond.cond_len_chars.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return 0

    bits_per = _bits_per_unit(cfg)

    n_units = getattr(cond_cfg, "cond_len_tokens", None)
    if n_units is None:
        n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
    else:
        n_units = int(n_units)

    cL = int(n_units * bits_per)
    return max(0, min(int(cL), int(seq_len_bits)))


def _sample_cond_len_bits_per_example(
    cfg, B: int, seq_len_bits: int, device
) -> torch.Tensor:
    """
    Returns cL_bits per example: [B] int64, in [0, seq_len_bits]. If
    cfg.cond.sample_prompt_len=False (default), returns fixed length
    repeated.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    if not sample_len:
        cL = _cond_len_bits_fixed(cfg, seq_len_bits)
        return torch.full((B,), int(cL), device=device, dtype=torch.long)

    bits_per = _bits_per_unit(cfg)

    # min/max in units (tokens or chars)
    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        # fallback to legacy names
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    # uniform integer in [mn, mx]
    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(
            low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long
        )

    cL_bits = units * int(bits_per)
    cL_bits = torch.clamp(cL_bits, min=0, max=int(seq_len_bits)).to(torch.long)
    return cL_bits


def _make_prefix_mask_from_lengths(
    cL_bits: torch.Tensor, S: int
) -> torch.Tensor:
    """
    cL_bits: [B] long returns prefix_mask: [B,S] bool where True
    indicates "prefix / conditioned" positions.
    """
    B = int(cL_bits.numel())
    ar = torch.arange(S, device=cL_bits.device).view(1, S).expand(B, S)
    return ar < cL_bits.view(B, 1)


def _make_null_value(
    cfg,
    device,
    dtype,
    *,
    is_tokens: bool = False,
    vocab_size: int | None = None,
) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    strategy = (
        str(getattr(cond_cfg, "null_strategy", "half"))
        if cond_cfg is not None
        else "half"
    )

    if is_tokens:
        if strategy in {"half", "data_center"}:
            if vocab_size is None:
                raise ValueError("vocab_size required for token null value")
            dc = float(
                getattr(
                    cfg.diffusion.continuous, "data_center", 1.0 / vocab_size
                )
            )
            return torch.tensor(dc, device=device, dtype=dtype)
        if strategy == "zeros":
            return torch.tensor(0.0, device=device, dtype=dtype)
        if strategy == "random":
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")

    if strategy == "half":
        return torch.tensor(0.5, device=device, dtype=dtype)
    if strategy == "data_center":
        return torch.tensor(
            float(getattr(cfg.diffusion.continuous, "data_center", 0.5)),
            device=device,
            dtype=dtype,
        )
    if strategy == "zeros":
        return torch.tensor(0.0, device=device, dtype=dtype)
    if strategy == "random":
        return torch.tensor(float("nan"), device=device, dtype=dtype)
    raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")


def _make_null_prefix_full(
    x0_full: torch.Tensor, prefix_mask: torch.Tensor, cfg
) -> torch.Tensor:
    """
    x0_full:
      - binary mode: [B,S]
      - token mode:  [B,S,V]
    prefix_mask: [B,S] bool
    """
    is_tokens = x0_full.dim() == 3
    b = x0_full.size(0)
    s = x0_full.size(1)

    cond_cfg = getattr(cfg, "cond", None)
    strategy = (
        str(getattr(cond_cfg, "null_strategy", "half"))
        if cond_cfg is not None
        else "half"
    )

    out = x0_full.clone()
    if not prefix_mask.any():
        return out

    if is_tokens:
        v = x0_full.size(-1)
        pm = prefix_mask.unsqueeze(-1).expand_as(x0_full)

        if strategy == "random":
            rnd = torch.full(
                (b, s, v), 1.0 / v, device=x0_full.device, dtype=x0_full.dtype
            )
            out[pm] = rnd[pm]
            return out

        null_val = _make_null_value(
            cfg, x0_full.device, x0_full.dtype, is_tokens=True, vocab_size=v
        )
        out[pm] = null_val
        return out

    if strategy == "random":
        rnd = torch.bernoulli(
            torch.full((b, s), 0.5, device=x0_full.device, dtype=x0_full.dtype)
        )
        out[prefix_mask] = rnd[prefix_mask]
        return out

    null_val = _make_null_value(
        cfg, x0_full.device, x0_full.dtype, is_tokens=False
    )
    out[prefix_mask] = null_val
    return out


def _discrete_positions_per_token(cfg) -> int:
    """
    Number of model positions corresponding to one semantic/BPE token in
    the discrete branch.

    - binary bitstream discrete: one token = bits_per_token model
    positions
    - token discrete: one token = one model position
    """
    repr_mode = str(getattr(cfg.data, "representation", "tokens")).lower()
    if repr_mode == "binary":
        return _bits_per_unit(cfg)
    return 1


def _sample_cond_len_positions_per_example_continuous(
    cfg, B: int, seq_len_positions: int, device
) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    repr_mode = str(
        getattr(getattr(cfg, "data", object()), "representation", "binary")
    ).lower()

    if repr_mode == "tokens":
        pos_per_unit = 1
    else:
        pos_per_unit = _bits_per_unit(cfg)

    if not sample_len:
        n_units = getattr(cond_cfg, "cond_len_tokens", None)
        if n_units is None:
            n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
        else:
            n_units = int(n_units)

        cL = max(0, min(int(n_units * pos_per_unit), int(seq_len_positions)))
        return torch.full((B,), cL, device=device, dtype=torch.long)

    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(
            low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long
        )

    cL = units * int(pos_per_unit)
    return torch.clamp(cL, min=0, max=int(seq_len_positions)).to(torch.long)


def _sample_cond_len_positions_per_example_discrete(
    cfg, B: int, seq_len_positions: int, device
) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    if not sample_len:
        n_units = getattr(cond_cfg, "cond_len_tokens", None)
        if n_units is None:
            n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
        else:
            n_units = int(n_units)
        pos_per_tok = _discrete_positions_per_token(cfg)
        cL = max(0, min(int(n_units * pos_per_tok), int(seq_len_positions)))
        return torch.full((B,), cL, device=device, dtype=torch.long)

    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(
            low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long
        )

    pos_per_tok = _discrete_positions_per_token(cfg)
    cL = units * int(pos_per_tok)
    return torch.clamp(cL, min=0, max=int(seq_len_positions)).to(torch.long)


def _dataloader_kwargs(cfg) -> dict:
    """
    DataLoader kwargs with version-safe handling of prefetch_factor and
    persistent_workers.

    Key rule: prefetch_factor is only valid when num_workers > 0.
    """
    nw = int(getattr(cfg.data, "num_workers", 0))
    pm = bool(getattr(cfg.data, "pin_memory", True))
    pf = int(getattr(cfg.data, "prefetch_factor", 2))

    kw = dict(
        num_workers=nw,
        pin_memory=pm,
    )
    if nw > 0:
        kw["prefetch_factor"] = pf
        # Persistent workers are incompatible with a scheduled task
        # mixture.
        #
        # The collate_fn is pickled into each worker when the worker
        # starts. With persistent_workers=True that happens ONCE for the
        # whole run, so nothing the trainer later writes to
        # `collate_fn.task_weights` -- the phase change of the (corpus,
        # task) mixture, and the per-epoch crop re-roll -- ever reaches
        # the process that actually collates. The mixture would silently
        # stay in phase A for 300,000 steps while every log reported
        # otherwise.
        #
        # Without it, workers are re-forked at each `iter(dataloader)`,
        # i.e. once per epoch, which is where the trainer applies those
        # updates. The respawn cost is amortised over 5,000 optimizer
        # steps.
        uses_schedule = bool(
            getattr(getattr(cfg, "data", None), "use_mixture_schedule", False)
            or str(
                getattr(
                    getattr(cfg, "data", None), "long_chain_policy", "drop"
                )
            )
            != "drop"
        )
        kw["persistent_workers"] = not uses_schedule
    return kw


def _resolve_checkpointing_cfg(cfg) -> dict:
    """
    Backward-compatible checkpointing config resolver.

    Priority:
      1) cfg.train.checkpointing.* (new)
      2) legacy cfg.train.save_last / save_top_k / checkpoint_mode
      3) legacy cfg.train.ckpt_interval.* (if present from earlier
      experiments)
      4) defaults

    New concept:
      - interval: sparse archival checkpoints (step=....pt)
      - resume_interval: frequent rolling checkpoint (last.pt only)
    """
    train = getattr(cfg, "train", None)

    # --- defaults ---
    save_last = True
    save_top_k = 1
    mode = "min"

    # archival interval checkpoints
    interval_enabled = False
    interval_every_steps = 100_000
    interval_keep_last = 5  # None/0 means keep all

    # rolling resume checkpoint
    resume_interval_enabled = False
    resume_interval_every_steps = 5_000

    # --- legacy: cfg.train.* ---
    if train is not None:
        if hasattr(train, "save_last"):
            save_last = bool(train.save_last)
        if hasattr(train, "save_top_k"):
            save_top_k = int(train.save_top_k)
        if hasattr(train, "checkpoint_mode"):
            mode = str(train.checkpoint_mode)

        # legacy interval block
        if hasattr(train, "ckpt_interval"):
            ci = train.ckpt_interval
            interval_enabled = bool(getattr(ci, "enabled", interval_enabled))
            interval_every_steps = int(
                getattr(ci, "every_steps", interval_every_steps)
            )
            interval_keep_last = getattr(ci, "keep_last", interval_keep_last)

    # --- new structured config overrides legacy ---
    if train is not None and hasattr(train, "checkpointing"):
        ck = train.checkpointing
        save_last = bool(getattr(ck, "save_last", save_last))
        save_top_k = int(getattr(ck, "save_top_k", save_top_k))
        mode = str(getattr(ck, "mode", mode))

        if hasattr(ck, "interval"):
            ci = ck.interval
            interval_enabled = bool(getattr(ci, "enabled", interval_enabled))
            interval_every_steps = int(
                getattr(ci, "every_steps", interval_every_steps)
            )
            interval_keep_last = getattr(ci, "keep_last", interval_keep_last)

        if hasattr(ck, "resume_interval"):
            ri = ck.resume_interval
            resume_interval_enabled = bool(
                getattr(ri, "enabled", resume_interval_enabled)
            )
            resume_interval_every_steps = int(
                getattr(ri, "every_steps", resume_interval_every_steps)
            )

    if mode not in {"min", "max"}:
        raise ValueError(
            f"checkpointing.mode must be 'min' or 'max', got {mode}"
        )

    # Interpret keep_last:
    #   None or 0 => keep all
    #   positive int => keep last N
    #   negative => treat as keep all
    if interval_keep_last is None:
        keep_last_norm = None
    else:
        try:
            keep_last_norm = int(interval_keep_last)
        except Exception:
            keep_last_norm = None
        if keep_last_norm is not None and keep_last_norm <= 0:
            keep_last_norm = None

    return {
        "save_last": save_last,
        "save_top_k": save_top_k,
        "mode": mode,
        "interval_enabled": interval_enabled,
        "interval_every_steps": interval_every_steps,
        "interval_keep_last": keep_last_norm,
        "resume_interval_enabled": resume_interval_enabled,
        "resume_interval_every_steps": resume_interval_every_steps,
    }


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # ── Distributed Setup ────────────────────────────────────────────────
        # Distinguish "requested" vs "active" to avoid crashes/hangs
        # when config and launcher disagree.
        self.ddp_requested = bool(getattr(cfg.system, "distributed", False))
        self.ddp_active = bool(self.ddp_requested and _ddp_is_on())

        if self.ddp_active:
            self.rank = int(cfg.system.global_rank)
            self.local_rank = int(cfg.system.local_rank)
            self.world_size = int(cfg.system.world_size)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            # Keep the old attribute name for minimal disruption /
            # compatibility.
        self.is_distributed = self.ddp_active
        self.is_master = self.rank == 0

        # ── Device selection
        # ─────────────────────────────────────────────────────
        # DDP: bind each process to cuda:<local_rank> Single GPU /
        # non-DDP: respect cfg.device (e.g. "cuda:1")
        if torch.cuda.is_available():
            if self.ddp_active:
                dev = torch.device(f"cuda:{self.local_rank}")
            else:
                dev_str = str(getattr(cfg, "device", "cuda:0"))
                # allow "cuda" as shorthand
                dev = torch.device("cuda:0" if dev_str == "cuda" else dev_str)

            if dev.type != "cuda":
                raise ValueError(
                    f"cfg.device must be a CUDA device when CUDA is "
                    f"available, got: {dev}"
                )

            torch.cuda.set_device(dev.index)
            self.device = dev
            self.cfg.device = str(dev)
        else:
            self.device = torch.device("cpu")
            self.cfg.device = "cpu"

        # Enable fast SDPA backends (flash/mem-efficient) on GPU.
        _enable_flash_sdp()
        if self.is_master:
            flash_avail = "Unknown (PyTorch < 2.3)"
            if torch.cuda.is_available() and hasattr(
                torch.backends.cuda, "is_flash_attention_available"
            ):
                flash_avail = (
                    torch.backends.cuda.is_flash_attention_available()
                )
            print(
                f"[FlashSDP] Enabled flash & mem-efficient attention "
                f"backends. flash_available={flash_avail}"
            )

        _maybe_set_seed(cfg)

        # ── dirs / writer (HPC-safe TB manager) ─────────────────────────────
        self.run_dir = (
            Path(getattr(cfg, "output_root", "runs")) / cfg.experiment
        )
        self.ckpt_dir = self.run_dir / "checkpoints"
        self._protebit_checkpoint_provenance = None
        if self.is_master:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_config_to_run_dir(cfg, self.run_dir)

        # DDP filesystem sync (robust on network FS)
        if self.ddp_active:
            dist.barrier()

        # TensorBoard manager + writer
        self.tb = None
        self.writer = _NullWriter()

        tb_cfg = getattr(getattr(cfg, "logging", None), "tensorboard", None)
        tb_enabled = (
            True if tb_cfg is None else bool(getattr(tb_cfg, "enabled", True))
        )

        if self.is_master and tb_enabled:
            self.tb = TBManager(
                cfg=cfg,
                run_dir=self.run_dir,
                subdir="training_logs",
                is_master=True,
            )

        if tb_cfg is None:
            self.tb_scalar_every_steps = 20
            self.tb_sync_every_steps = 0
            self.tb_sync_every_epochs = 1
        else:
            self.tb_scalar_every_steps = int(
                getattr(tb_cfg, "scalar_every_steps", 20)
            )
            self.tb_sync_every_steps = int(
                getattr(tb_cfg, "sync_every_steps", 0)
            )
            self.tb_sync_every_epochs = int(
                getattr(tb_cfg, "sync_every_epochs", 1)
            )

        # ── W&B init (optional) ─────────────────────────────────────────────
        logging_cfg = getattr(cfg, "logging", None)
        self.use_wandb: bool = False
        # When True, every TensorBoard write (scalars, figures, images,
        # histograms, text) from the trainer AND all callbacks is
        # mirrored to W&B automatically, so a run is fully readable in
        # W&B without a per-metric _log_wandb call. In that mode the
        # explicit
        # _log_wandb calls
        # stand down (see _log_wandb) to avoid duplicate,
        # step-conflicting series.
        self._wandb_sync_tb: bool = False
        # Running (EMA) smoothing of the noisy per-step train loss for
        # logging.
        self._loss_ema: float | None = None

        # Throughput telemetry. Without these there is no way to tell a
        # slow interconnect from a slow data loader from a slow
        # checkpoint save, and every utilisation claim about this model
        # is unfalsifiable: the only step rate anyone had was inferred
        # afterwards from TensorBoard wall times, which bundles
        # validation and the task gate into the training rate. `wait` is
        # the share of wall time the training loop spends blocked on the
        # loader -- the single number that says whether more dataloader
        # workers would buy anything.
        self._tp_prev_end: float | None = None
        self._tp_wait = 0.0
        self._tp_busy = 0.0
        self._tp_tokens = 0.0
        self._tp_optim_steps = 0
        # Link-wide counters used by the 1B bootstrap probes. Unlike the
        # interval counters above these are never reset by TensorBoard
        # logging.
        self._run_tp_wait = 0.0
        self._run_tp_busy = 0.0
        self._run_tp_tokens = 0.0
        self._run_tp_optim_steps = 0
        self._run_tp_start_step = 0
        self._run_last_train_loss: float | None = None
        self._run_first_train_loss: float | None = None
        self._run_last_val_loss: float | None = None

        # WANDB_MODE=disabled/dryrun is a hard off, honoured regardless
        # of config.
        _wandb_hard_off = os.environ.get("WANDB_MODE", "").lower() in (
            "disabled",
            "dryrun",
        )
        if (
            self.is_master
            and logging_cfg is not None
            and getattr(logging_cfg, "use_wandb", False)
            and not _wandb_hard_off
        ):
            if WANDB_AVAILABLE:
                self.use_wandb = True
                os.environ["WANDB_PYTORCH_DISABLE"] = "true"
                os.environ["WANDB_DISABLE_GRADIENTS"] = "true"

                cfg_dict = _cfg_to_dict(cfg)
                # Environment variables win over config so a run can be
                # redirected or switched to offline without editing the
                # config.
                project = os.environ.get("WANDB_PROJECT") or getattr(
                    logging_cfg, "project", "diffusion"
                )
                entity = os.environ.get("WANDB_ENTITY") or getattr(
                    logging_cfg, "entity", None
                )
                mode = os.environ.get("WANDB_MODE") or getattr(
                    logging_cfg, "mode", "online"
                )
                group = getattr(logging_cfg, "group", None)
                tags = getattr(logging_cfg, "tags", None)
                # W&B display name defaults to the experiment (= run
                # dir), but can be set independently via
                # cfg.logging.run_name or WANDB_NAME.
                run_name = (
                    os.environ.get("WANDB_NAME")
                    or getattr(logging_cfg, "run_name", None)
                    or cfg.experiment
                )
                # Default on: mirror the full TensorBoard stream into
                # W&B. The SafeSummaryWriter is created after this init
                # (in prepare_for_run), so wandb's writer patch is
                # active in time.
                self._wandb_sync_tb = bool(
                    getattr(logging_cfg, "sync_tensorboard", True)
                )

                # A stable id is what makes a 12-link chain ONE W&B
                # curve instead of twelve. Derived from the experiment
                # name so it is reproducible without any launcher state,
                # and overridable by WANDB_RUN_ID (which common.sh sets
                # from the run name) or by the config. Passing id=None
                # would leave wandb to invent a new id on every link and
                # the run would appear as a fresh curve starting at step
                # 0 after every preemption.
                run_id = (
                    getattr(logging_cfg, "run_id", None)
                    or os.environ.get("WANDB_RUN_ID")
                    or hashlib.md5(
                        str(cfg.experiment).encode("utf-8")
                    ).hexdigest()[:16]
                )
                wandb.init(
                    project=project,
                    entity=entity,
                    id=run_id,
                    resume="allow",
                    name=run_name,
                    group=group,
                    tags=list(tags) if tags else None,
                    config=cfg_dict,
                    dir=str(self.run_dir),
                    mode=mode,
                    sync_tensorboard=self._wandb_sync_tb,
                )
                if wandb.run is not None:
                    print(f"W&B run: {wandb.run.get_url()}")
                    if self._wandb_sync_tb:
                        print(
                            "W&B: mirroring all TensorBoard "
                            "metrics/figures/images into this run "
                            "(sync_tensorboard=True)."
                        )
                    # Persist the run identity so a later eval process
                    # can resume this exact run and attach eval metrics
                    # to it (see utils/wandb_eval.log_eval_metrics).
                    try:
                        (self.run_dir / "wandb_run.json").write_text(
                            json.dumps(
                                {
                                    "id": wandb.run.id,
                                    "entity": wandb.run.entity,
                                    "project": wandb.run.project,
                                    "name": wandb.run.name,
                                    "url": wandb.run.get_url(),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
            else:
                print("⚠️  wandb not installed, skipping W&B logging.")

        # ── checkpoint config / bookkeeping (resolved +
        # backward-compatible) ─────
        ck = _resolve_checkpointing_cfg(cfg)

        self.save_last = bool(ck["save_last"])
        self.save_top_k = int(ck["save_top_k"])
        self.checkpoint_mode = str(ck["mode"])

        self.best_metric = (
            math.inf if self.checkpoint_mode == "min" else -math.inf
        )
        self.best_ckpts: list[dict] = []
        self.early_stop_best = self.best_metric
        self.early_stop_bad_epochs = 0

        # periodic interval checkpoints (post-training analysis)
        self.ckpt_interval_enabled = bool(ck["interval_enabled"])
        self.ckpt_interval_every_steps = int(ck["interval_every_steps"])
        self.ckpt_interval_keep_last = ck[
            "interval_keep_last"
        ]  # None => keep all
        self.resume_interval_enabled = bool(ck["resume_interval_enabled"])
        self.resume_interval_every_steps = int(
            ck["resume_interval_every_steps"]
        )

        # interval state (final threshold set after resume)
        self._next_interval_ckpt_step = None
        self._next_resume_ckpt_step = None
        self._interval_ckpt_paths: list[str] = []

        # ── walltime deadline / preemption ──────────────────────────────────
        # EOS caps every job at 4 hours, so a real run is a chain of
        # links that each resume from last.pt. The guard makes the job
        # save and exit before Slurm kills it, and makes every rank
        # agree on when that happens.
        train_cfg = getattr(cfg, "train", None)
        self._deadline = DeadlineGuard(
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            margin_seconds=getattr(train_cfg, "exit_margin_seconds", None),
            max_wall_seconds=getattr(train_cfg, "max_wall_seconds", None),
            check_every_steps=int(
                getattr(train_cfg, "preempt_check_every_steps", 20)
            ),
            verbose=self.is_master,
        )
        # Wall-clock checkpoint cadence. At a 4-hour walltime a
        # step-based cadence is the wrong unit: it saves too often early
        # and not often enough once the step time grows.
        self.resume_interval_every_seconds = float(
            getattr(train_cfg, "checkpoint_every_seconds", 900) or 0
        )
        self._last_resume_ckpt_time = time.time()

        # ── data ────────────────────────────────────────────────────────────
        dl_kw = _dataloader_kwargs(cfg)

        dataset_name = str(getattr(cfg.data, "dataset", "")).lower()
        is_dima_length_bucketed = dataset_name in {
            "swissprotdima",
            "swissprot_dima",
            "dima_swissprot",
        }
        is_evodiff_length_bucketed = dataset_name in {
            "evodiffuniref50",
            "evodiff_uniref50",
            "uniref50_evodiff",
        }
        is_multimodal = dataset_name in {
            "proteinmultimodallfq",
            "protein_multimodal",
            "dplm_paired",
        }
        # Multimodal state (kept harmless / off for every non-multimodal
        # path).
        self.is_multimodal = bool(is_multimodal)
        self.replay_fraction = 0.0
        self.replay_loader = None
        self._replay_iter = None
        self._replay_batches_drawn_epoch = 0
        self._mm_resume_replay_cursor = 0
        self.replay_exact_window = int(
            getattr(cfg.data, "sequence_replay_exact_window", 0) or 0
        )
        self.replay_exact_count = int(
            getattr(cfg.data, "sequence_replay_exact_count", 0) or 0
        )
        if (self.replay_exact_window == 0) != (self.replay_exact_count == 0):
            raise ValueError(
                "sequence replay exact_window and exact_count must both "
                "be set or both be zero"
            )
        if self.replay_exact_window and not (
            0 < self.replay_exact_count < self.replay_exact_window
        ):
            raise ValueError(
                "sequence replay exact_count must be in [1, exact_window)"
            )

        # ── unified (corpus, task) mixture ──────────────────────────────────
        # P0-6. The configured mixture and the realized mixture used to
        # be different objects. `mixture_schedule` makes the configured
        # numbers BE the realized step proportions, and the counters
        # below make that checkable rather than asserted.
        self.mixture_schedule = None
        self._mm_active_phase = -1
        self._mm_mix_counts: dict[str, int] = {}
        self._mm_mix_total = 0
        self._mm_corpus_microbatches = 0
        self._mm_replay_microbatches = 0
        self._mm_mix_logged_step = -1
        # Task-metric selection gate (P0-5). Built lazily on the master,
        # then cached, because freezing the subset costs a pass over the
        # validation set and the subset must not change between calls.
        self._task_scorer = None
        self._task_score_last_step = None
        self._task_score_error = None
        if bool(
            getattr(getattr(cfg, "data", None), "use_mixture_schedule", False)
        ):
            from data.protein_tasks import TaskMixtureSchedule

            phases = getattr(cfg.data, "mixture_phases", None)
            self.mixture_schedule = TaskMixtureSchedule(
                _parse_mixture_phases(phases) if phases is not None else None
            )
        self._last_mm_components = None
        self._last_mm_val_diagnostics = None

        if not (
            is_dima_length_bucketed
            or is_evodiff_length_bucketed
            or is_multimodal
        ):
            raw_train_loader, raw_val_loader, _ = get_dataloaders(cfg)

        if is_multimodal:
            # Keep the paired multimodal loaders exactly as built: the
            # custom source/length batch sampler and the task-aware
            # collator (which yields dict batches with x0/states/target
            # masks) must survive, so never rebuild a plain DataLoader
            # from `.dataset` here.
            from data.protein_multimodal import get_multimodal_loader

            if self.ddp_active:
                assert cfg.train.batch_size % self.world_size == 0, (
                    f"Global batch_size ({cfg.train.batch_size}) must be "
                    f"divisible by world_size ({self.world_size})."
                )
                local_batch_size = cfg.train.batch_size // self.world_size
            else:
                local_batch_size = int(cfg.train.batch_size)
            mm_seed = int(getattr(cfg.train, "seed", 42))
            self.train_loader = get_multimodal_loader(
                cfg,
                split="train",
                batch_size=local_batch_size,
                shuffle=True,
                seed=mm_seed,
            )
            self.val_loader = get_multimodal_loader(
                cfg,
                split="val",
                batch_size=local_batch_size,
                shuffle=False,
                seed=mm_seed,
            )
            # Optional token-budgeted sequence-only UniRef50 replay
            # (plan 6.6 step 3). A separate paired-format shard set of
            # sequence-only rows (struct_mask all False) mixed by
            # `sequence_replay_fraction`; batches are the same dict
            # format so the multimodal step consumes them transparently.
            self.replay_fraction = float(
                getattr(cfg.data, "sequence_replay_fraction", 0.0) or 0.0
            )
            replay_root = str(
                getattr(cfg.data, "sequence_replay_root", "") or ""
            )
            replay_ready = (
                bool(replay_root)
                and (Path(replay_root) / "manifest.json").exists()
            )
            replay_required = bool(
                getattr(cfg.data, "require_sequence_replay", False)
            )
            if (
                self.replay_fraction > 0.0
                and replay_required
                and not replay_ready
            ):
                raise RuntimeError(
                    f"required sequence replay corpus is unavailable: "
                    f"expected manifest.json under {replay_root!r}"
                )
            if self.replay_fraction > 0.0 and replay_ready:
                # Build the replay loader over the sequence-only corpus.
                # The corpus has a single source ("uniref50") that is
                # absent from the paired source_weights, so pass
                # source_weights=None (equal weighting over present
                # sources) to avoid a zero-total-weight sampler error.
                # Force a sequence-producing task mix so every replay
                # example supervises its sequence bits (structure is
                # ABSENT on these rows, so paired or structure-target
                # tasks would waste the replay batch).
                orig_shard = cfg.data.shard_dir
                orig_manifest_sha = getattr(
                    cfg.data, "expected_manifest_sha256", ""
                )
                orig_src_w = getattr(cfg.data, "source_weights", None)
                orig_task_w = getattr(cfg.data, "task_weights", None)
                cfg.data.shard_dir = replay_root
                cfg.data.expected_manifest_sha256 = str(
                    getattr(
                        cfg.data,
                        "sequence_replay_expected_manifest_sha256",
                        "",
                    )
                    or ""
                )
                cfg.data.source_weights = None
                cfg.data.task_weights = _mm_replay_task_weights(cfg)
                try:
                    self.replay_loader = get_multimodal_loader(
                        cfg,
                        split="train",
                        batch_size=local_batch_size,
                        shuffle=True,
                        seed=mm_seed + 101,
                    )
                finally:
                    cfg.data.shard_dir = orig_shard
                    cfg.data.expected_manifest_sha256 = orig_manifest_sha
                    cfg.data.source_weights = orig_src_w
                    cfg.data.task_weights = orig_task_w
            elif self.replay_fraction > 0.0 and self.is_master:
                reason = (
                    "no cfg.data.sequence_replay_root set"
                    if not replay_root
                    else (
                        f"no manifest.json under sequence_replay_root "
                        f"{replay_root!r} (build it with "
                        f"scripts/proteins/setup/prepare_uniref50_rep"
                        f"lay.py)"
                    )
                )
                print(
                    f"[multimodal] sequence_replay_fraction>0 but {reason}"
                    f"; replay disabled."
                )
                self.replay_fraction = 0.0
            elif self.replay_fraction > 0.0:
                # Non-master ranks must agree on the disabled state to
                # stay in sync.
                self.replay_fraction = 0.0

            # A scheduled mixture that declares replay mass and has no
            # replay loader is a silently wrong recipe, not a degraded
            # one.
            #
            # Before the (corpus, task) schedule, a missing replay
            # corpus meant "replay off", which a reader could see in the
            # warning above. Now the mixture declares a replay SHARE --
            # 15-20% of every step -- and dropping it re-weights every
            # other task by 1/(1-share) without changing a single
            # configured number. The run would complete normally and
            # every mixture claim in the paper would be wrong.
            #
            # The run-end realized-vs-configured report catches it, but
            # only after the GPU-hours are spent. Fail at startup
            # instead.
            if (
                self.mixture_schedule is not None
                and self.replay_loader is None
            ):
                from data.protein_tasks import REPLAY_CORPUS

                declared = max(
                    self.mixture_schedule.corpus_fraction(p, REPLAY_CORPUS)
                    for p in (0.0, 0.5, 1.0)
                )
                if declared > 0.0:
                    raise RuntimeError(
                        f"cfg.data.use_mixture_schedule is on and the "
                        f"mixture declares up to {declared:.0%} replay, "
                        f"but no replay corpus was loaded from "
                        f"cfg.data.sequence_replay_root="
                        f"{getattr(cfg.data, 'sequence_replay_root', '')!r}"
                        f".\nBuild it first:\n  .venv/bin/python "
                        f"scripts/proteins/setup/prepare_uniref50_rep"
                        f"lay.py --out-dir <that path>\nOr set "
                        f"cfg.data.use_mixture_schedule = False to run the"
                        f" paired corpus alone, which is a different "
                        f"recipe and should be named as one."
                    )
        elif is_dima_length_bucketed or is_evodiff_length_bucketed:
            if self.ddp_active:
                assert cfg.train.batch_size % self.world_size == 0, (
                    f"Global batch_size ({cfg.train.batch_size}) must be "
                    f"divisible by world_size ({self.world_size})."
                )
                local_batch_size = cfg.train.batch_size // self.world_size
            else:
                local_batch_size = int(cfg.train.batch_size)
            protocol_loader = (
                get_evodiff_uniref50_loader
                if is_evodiff_length_bucketed
                else get_dima_loader
            )
            self.train_loader = protocol_loader(
                cfg,
                split="train",
                batch_size=local_batch_size,
                shuffle=True,
                seed=int(getattr(cfg.train, "seed", 42)),
            )
            self.val_loader = protocol_loader(
                cfg,
                split="val",
                batch_size=local_batch_size,
                shuffle=False,
                seed=int(getattr(cfg.train, "seed", 42)),
            )
        elif self.ddp_active:
            assert cfg.train.batch_size % self.world_size == 0, (
                f"Global batch_size ({cfg.train.batch_size}) must be "
                f"divisible by world_size ({self.world_size}) for "
                f"fixed-shape DDP training."
            )
            batch_size_per_gpu = cfg.train.batch_size // self.world_size

            train_sampler = DistributedSampler(
                raw_train_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.train_loader = torch.utils.data.DataLoader(
                raw_train_loader.dataset,
                batch_size=batch_size_per_gpu,
                sampler=train_sampler,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )

            val_sampler = DistributedSampler(
                raw_val_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=True,
            )
            self.val_loader = torch.utils.data.DataLoader(
                raw_val_loader.dataset,
                batch_size=batch_size_per_gpu,
                sampler=val_sampler,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                raw_train_loader.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                drop_last=True,
                **dl_kw,
            )
            self.val_loader = torch.utils.data.DataLoader(
                raw_val_loader.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )

        # ── model ───────────────────────────────────────────────────────────
        base = create_model(cfg).to(self.device)

        # IMPORTANT: Precompute Flex block masks BEFORE DDP wrapping /
        # torch.compile so create_block_mask(...) is never reached
        # inside compiled forward.
        raw_base_for_prep = _unwrap_all(base)
        if hasattr(raw_base_for_prep, "prepare_flex_masks"):
            if self.is_master:
                print(
                    "[flex] precomputing block masks before torch.compile ..."
                )
            raw_base_for_prep.prepare_flex_masks(self.device)

        if self.ddp_active:
            try:
                base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
            except Exception:
                pass
            # find_unused_parameters is required by the MULTIMODAL PATH
            # itself, not by the warm-start freeze window.
            #
            # It used to be keyed on `warm_start_freeze_trunk_steps >
            # 0`, on the reasoning that the freeze is what stops
            # parameters producing gradients. That reasoning was
            # incomplete, and the coupling hid the real requirement:
            # when the freeze was disabled (because the warm start was a
            # no-op and had to go), all four chain links died within 84
            # seconds with "Expected to have finished reduction in the
            # prior iteration", on a code path that had trained 5,000
            # steps happily the day before.
            #
            # The underlying reason is that this model's autograd graph
            # is DATA-DEPENDENT: every rank draws its own task mixture,
            # and a task decides which modality carries loss targets,
            # whether a span is observed or absent, and whether the CFG
            # condition was dropped. Ranks therefore do not agree on
            # which parameters participate in a given step. That is
            # inherent to a multi-task masked-diffusion model, not a
            # transient of the warm-start curriculum.
            #
            # NOT ROOT-CAUSED: a single-process full-config
            # forward/backward gives gradients to all 288 parameters, so
            # the divergence only appears across ranks and is not
            # reproducible on one. TORCH_DISTRIBUTED_DEBUG is set to
            # DETAIL in the launcher so the next occurrence names the
            # parameters instead of printing bucket indices.
            #
            # Cost: 5-10% throughput (plan section 9.2). Worth measuring
            # against a static-graph fix once the offending parameter is
            # known.
            mm_unused = bool(self.is_multimodal)
            if mm_unused and self.is_master:
                print(
                    "[ddp] find_unused_parameters=True (multimodal task "
                    "mixture gives each rank a data-dependent graph; "
                    "costs 5-10% throughput -- see trainers/trainer.py "
                    "for why)"
                )
            # device_ids/output_device are only valid for a
            # single-CUDA-device module; DDP rejects them outright on
            # CPU. Passing them unconditionally is why this trainer
            # could never be exercised under a real process group on a
            # CPU box -- and every one of the three consecutive 8-GPU
            # failures lived in code that only runs with world_size > 1.
            # Making the wrap CPU-capable is what buys a gloo-backed
            # end-to-end test (tests/test_distributed_smoke.py).
            # gradient_as_bucket_view makes each parameter's .grad a
            # view into the DDP reduction bucket instead of a separate
            # tensor, so the gradient copy into the bucket disappears
            # and one full copy of the gradients stops being allocated.
            # It changes no arithmetic -- the reduced values are
            # identical -- so unlike a batch-size change it costs
            # nothing scientifically. static_graph, the other usual
            # companion, is NOT set: it is incompatible with the
            # data-dependent graph documented above, which is the same
            # reason find_unused_parameters has to be on.
            ddp_kwargs = {
                "find_unused_parameters": mm_unused,
                "gradient_as_bucket_view": True,
            }
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            base = DDP(base, **ddp_kwargs)

        compile_enabled = bool(getattr(self.cfg.train, "use_compile", False))
        compile_mode = getattr(self.cfg.train, "compile_mode", "default")

        if compile_enabled and hasattr(torch, "compile"):
            try:
                if self.is_master:
                    print(
                        f"[torch.compile] compiling model with mode="
                        f"{compile_mode!r}..."
                    )
                base = torch.compile(base, mode=compile_mode, fullgraph=False)
                if self.is_master:
                    print("[torch.compile] done.")
            except Exception as e:
                print(
                    f"[torch.compile] WARNING: failed to compile model ("
                    f"{e}); using eager mode."
                )
        else:
            if not hasattr(torch, "compile") and self.is_master:
                print(
                    "[torch.compile] not available in this PyTorch; "
                    "using eager mode."
                )

        self.model = base

        # ── P0-4: per-rank stochastic stream ───────────────────────────────
        # Weight init above used the shared seed, so every rank holds
        # identical parameters. From here on each rank needs its OWN
        # noise stream, or the whole world draws the same sigmas and the
        # same Gaussian noise and a 64-GPU batch carries only
        # micro_batch_per_gpu distinct noise levels.
        self._stochastic_seed = reseed_stochastic_stream(cfg, self.rank)
        if self.is_master and self._stochastic_seed >= 0:
            print(
                f"[seed] shared init seed="
                f"{getattr(cfg.train, 'seed', None)}; per-rank stochastic "
                f"seed on rank 0 = {self._stochastic_seed}"
            )

        # ── gradient accumulation ──────────────────────────────────────────
        # Accumulate this many micro-batches into one optimizer step so
        # the effective global batch (in residue-patches) is held
        # constant as the node/GPU count changes. 1 == the previous
        # single-step behavior.
        self.grad_accum_steps = max(
            1, int(getattr(cfg.train, "grad_accum_steps", 1) or 1)
        )
        self._accum_counter = 0
        self._did_optim_step = True
        # Optimizer steps completed in the current epoch (P0-7). Reset
        # at each epoch start; kept separate from _epoch_batches_done,
        # which counts micro-batches and is the sampler resume cursor.
        self._epoch_optim_steps_done = 0

        # ── multimodal warm-start / curriculum knobs (gate 3) ──────────────
        # Freezing the trunk is a warm-start curriculum step: it
        # protects transferred weights while the new adapters catch up.
        # Freezing weights that were NOT transferred is strictly harmful
        # -- it spends the freeze window training 6M of 126M parameters
        # at the start of the LR warmup, and it pins DDP
        # find_unused_parameters on for the whole run at a 5-10%
        # throughput cost. _mm_warm_start clears this when the transfer
        # does not clear the floor.
        self.freeze_trunk_steps = int(
            getattr(cfg.model, "warm_start_freeze_trunk_steps", 0) or 0
        )
        self._warm_start_fraction = None
        self.trunk_lr_mult = float(
            getattr(cfg.model, "warm_start_trunk_lr_mult", 1.0) or 1.0
        )
        self.warm_start_seq_checkpoint = str(
            getattr(cfg.model, "warm_start_seq_checkpoint", "") or ""
        )
        self._trunk_frozen = False
        # A dedicated lower-LR trunk group is only worth its resume
        # complexity when the multimodal path actually asks for a
        # different trunk LR.
        self._trunk_group_active = bool(
            self.is_multimodal and abs(self.trunk_lr_mult - 1.0) > 1e-9
        )

        # ── opt / ema / amp ────────────────────────────────────────────────
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)
        ema_ramp_cfg = getattr(cfg.train, "ema_ramp", None)
        self.ema_ramp_enabled = bool(
            getattr(ema_ramp_cfg, "enabled", False)
            if ema_ramp_cfg is not None
            else False
        )
        self.ema_ramp_start = float(
            getattr(ema_ramp_cfg, "start_decay", cfg.train.ema_decay)
            if ema_ramp_cfg is not None
            else cfg.train.ema_decay
        )
        self.ema_ramp_end = float(
            getattr(ema_ramp_cfg, "end_decay", cfg.train.ema_decay)
            if ema_ramp_cfg is not None
            else cfg.train.ema_decay
        )
        self.ema_ramp_steps = int(
            getattr(ema_ramp_cfg, "steps", 0)
            if ema_ramp_cfg is not None
            else 0
        )
        if self.ema_ramp_enabled:
            self.ema.decay = _ema_decay_for_step(
                self.ema_ramp_start, self.ema_ramp_end, self.ema_ramp_steps, 0
            )
        if self._trunk_group_active:
            # Group 0 = new multimodal adapters (base LR); group 1 =
            # warm-started trunk (base LR, scaled down each step by
            # trunk_lr_mult). Built here (before resume) so a resumed
            # optimizer state matches the structure.
            self.opt, self.lr_sched = get_optimizer_and_scheduler(
                self.model, cfg, 0, params=self._mm_build_param_groups()
            )
        else:
            self.opt, self.lr_sched = get_optimizer_and_scheduler(
                self.model, cfg, 0
            )

        self.amp_enabled = bool(getattr(cfg.train, "use_fp16", False))
        amp_dtype_req = str(getattr(cfg.train, "amp_dtype", "auto")).lower()

        if not self.amp_enabled:
            self.amp_dtype = torch.float32
        else:
            if amp_dtype_req in {"bf16", "bfloat16"}:
                self.amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            elif amp_dtype_req in {"fp16", "float16"}:
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )

        self.use_scaler = (
            self.amp_enabled
            and (self.device.type == "cuda")
            and (self.amp_dtype == torch.float16)
        )
        self.scaler = GradScaler(enabled=self.use_scaler)
        self.grad_clip = float(getattr(cfg.optim, "grad_clip", 1.0))

        # ──────────────────────────────────────────────────────────────────
        # Entropy modes: none vs online(adaptive) vs offline
        # ──────────────────────────────────────────────────────────────────
        off = getattr(cfg.train, "entropy_offline", None)
        self.entropy_offline_enabled = (
            bool(getattr(off, "enabled", False)) if off is not None else False
        )

        legacy_entropy_online = bool(
            getattr(cfg.train, "entropy_online", False)
        )
        entropy_compute_cfg = getattr(cfg.train, "entropy_compute", None)
        entropy_use_cfg = getattr(cfg.train, "entropy_use_for_sampling", None)

        if entropy_compute_cfg is None and entropy_use_cfg is None:
            self.entropy_compute = legacy_entropy_online
            self.entropy_use_for_sampling = legacy_entropy_online
        else:
            self.entropy_compute = (
                bool(entropy_compute_cfg)
                if entropy_compute_cfg is not None
                else legacy_entropy_online
            )
            self.entropy_use_for_sampling = (
                bool(entropy_use_cfg)
                if entropy_use_cfg is not None
                else legacy_entropy_online
            )

        if (
            self.entropy_use_for_sampling
            and not self.entropy_compute
            and not self.entropy_offline_enabled
        ):
            raise ValueError(
                "entropy_use_for_sampling=True but "
                "entropy_compute=False.\nEnable entropy_compute for "
                "online adaptive scheduling, or enable "
                "cfg.train.entropy_offline.enabled=True for offline "
                "scheduling."
            )

        # aliases
        self.entropy_online = self.entropy_use_for_sampling
        if self.entropy_offline_enabled:
            self.entropy_profile_source = "offline"
        elif self.entropy_compute:
            self.entropy_profile_source = "online"
        else:
            self.entropy_profile_source = "none"

        # hyperparams
        self.entropy_buffer_size = int(
            getattr(cfg.train, "entropy_buffer_size", 100_000)
        )
        self.entropy_num_bins = int(
            getattr(cfg.train, "entropy_num_bins", 256)
        )
        self.entropy_warmup_steps = int(
            getattr(cfg.train, "entropy_warmup_steps", 100_000)
        )
        self.entropy_transition_steps = int(
            getattr(cfg.train, "entropy_transition_steps", 100_000)
        )
        self.entropy_gamma_max = float(
            getattr(cfg.train, "entropy_gamma_max", 0.5)
        )

        # ✅ Gabriel regularized schedule knobs
        self.entropy_mode = str(
            getattr(cfg.train, "entropy_mode", "regularized")
        ).lower()
        self.entropy_regularizer_c = float(
            getattr(cfg.train, "entropy_regularizer_c", 0.1)
        )
        self.entropy_regularizer_n = float(
            getattr(cfg.train, "entropy_regularizer_n", 3.0)
        )

        # ✅ rate vs sqrt-rate via exponent p
        target = getattr(cfg.train, "entropy_target", "rate")
        if isinstance(target, str):
            t = target.lower()
            if t == "rate":
                self.entropy_rate_power = 1.0
            elif t in {"sqrt", "sqrt_rate", "sqrt-rate"}:
                self.entropy_rate_power = 0.5
            else:
                self.entropy_rate_power = float(t)  # allow "0.75" etc
        else:
            self.entropy_rate_power = float(target)

        self.entropy_target = target

        # ✅ FIFO ring buffer (CPU) — stable memory, no growth
        cap = self.entropy_buffer_size
        self._entropy_sig_buf = torch.empty(
            cap, dtype=torch.float32, device="cpu"
        )
        self._entropy_metric_buf = torch.empty(
            cap, dtype=torch.float32, device="cpu"
        )
        self._entropy_buf_ptr = 0
        self._entropy_buf_len = 0

        # entropy tables / state
        self._entropy_ready = False
        self._entropy_pdf = None
        self._entropy_cdf = None
        self._entropy_sigmas = None
        self._entropy_edges = None
        self._entropy_ln_mu = None
        self._entropy_ln_std = None

        self.callbacks: list[Callback] = []

        # ── framework-specific components ───────────────────────────────────
        if cfg.framework == "continuous_score":
            self.proc = ContinuousForwardProcess(cfg)
            self.entropy_ctrl = EntropyScheduleController(self, self.proc)
            repr_mode = str(
                getattr(self.cfg.data, "representation", "binary")
            ).lower()
            if repr_mode == "tokens":
                self.loss_fn = token_score_interpolation_loss
            else:
                self.loss_fn = binary_score_interpolation_loss
            solver_name = str(
                getattr(getattr(cfg, "evaluation", object()), "solver", "heun")
            ).lower()
            if solver_name in {"ddim", "ddim_entropic"}:
                self.sampler = DDIMSampler(self.model, self.proc, cfg)
            elif solver_name in {"heun", "heun_entropic"}:
                self.sampler = HeunSampler(self.model, self.proc, cfg)
            else:
                raise ValueError(f"unknown continuous solver {solver_name!r}")

            # ── callbacks ─────────────────────────────────────────────
            # ───────────────

            # Offline entropy profile: if it uses dist collectives, keep
            # it on all ranks.
            if self.entropy_offline_enabled:
                self.callbacks.append(OfflineEntropyProfileCallback(cfg))

            # SigmaDataEstimator / plotting are master-only (no
            # collectives, pure logging). They assume single-tensor
            # sequence batches and a single scalar-sigma schedule, so
            # they are skipped on the multimodal path (dict batches,
            # independent modality noise; sigma_data is configured).
            if self.is_master and not self.is_multimodal:
                self.callbacks.append(SigmaDataEstimator(num_batches=10))
                self.callbacks.append(
                    EntropySchedulePlotCallback(
                        every_k_epochs=int(
                            getattr(
                                cfg.train, "entropy_plot_every_k_epochs", 20
                            )
                        )
                    )
                )

            # ---- ALL-RANK callbacks that use dist collectives ----
            # External PPL
            ext_cfg = getattr(cfg.train, "external_perplexity", None)
            if ext_cfg is None:
                ext_cfg = getattr(cfg.train, "external_ppl", None)

            if ext_cfg is not None and bool(
                getattr(ext_cfg, "enabled", False)
            ):
                self.callbacks.append(ExternalPPLCallback(cfg))

            # VLB (All ranks - critical for DDP synchronization).
            # Skipped on the multimodal path: VLBBoundCallback rebuilds
            # the eval loader without the MultimodalTaskCollator, so the
            # dict rows are default-collated and compute_vlb_over_loader
            # crashes calling .to(device) on a dict. The per-modality
            # loss/accuracy are logged directly from the multimodal step
            # instead, so no VLB estimate is lost that the multimodal
            # path produces.
            vlb_cfg = getattr(cfg.train, "vlb", None)
            if (
                vlb_cfg is not None
                and bool(getattr(vlb_cfg, "enabled", False))
                and not self.is_multimodal
            ):
                self.callbacks.append(
                    VLBBoundCallback(
                        every_k_epochs=int(
                            getattr(vlb_cfg, "every_k_epochs", 10)
                        ),
                        sigma_min_eval=getattr(
                            vlb_cfg, "sigma_min_eval", None
                        ),
                        sigma_max_eval=getattr(
                            vlb_cfg, "sigma_max_eval", None
                        ),
                        sigma_sampling=getattr(
                            vlb_cfg, "sigma_sampling", "log-uniform"
                        ),
                        num_mc_samples_per_batch=int(
                            getattr(vlb_cfg, "num_mc_samples_per_batch", 1)
                        ),
                        include_prior=bool(
                            getattr(vlb_cfg, "include_prior", False)
                        ),
                        use_amp=bool(getattr(vlb_cfg, "use_amp", True)),
                        progress=bool(getattr(vlb_cfg, "progress", False)),
                    )
                )

            # MAUVE
            mauve_cfg = getattr(cfg.train, "mauve", None)
            if mauve_cfg is not None and bool(
                getattr(mauve_cfg, "enabled", False)
            ):
                self.callbacks.append(MauveCallback(cfg))  # pass FULL cfg

            # Visualization / sample dumps
            vis_cfg = getattr(cfg.train, "visualization", None)
            if vis_cfg is not None and bool(
                getattr(vis_cfg, "enabled", False)
            ):
                self.callbacks.append(VisualizationCallback(cfg))

        elif cfg.framework == "discrete_sedd":
            self.proc = DiscreteForwardProcess(cfg)
            self.loss_fn = dwdse_loss

            repr_mode = str(
                getattr(cfg.data, "representation", "tokens")
            ).lower()
            if repr_mode == "binary":
                seq_len = int(getattr(cfg.data, "sequence_len", 0))
            else:
                seq_len = int(
                    getattr(
                        cfg.data,
                        "sequence_len_tokens",
                        getattr(
                            cfg.data,
                            "sequence_len_chars",
                            getattr(cfg.data, "sequence_len", 0),
                        ),
                    )
                )

            self.sampler = TweedieTauLeapingSampler(
                model=self.model,
                process=self.proc,
                device=self.device,
                vocab_size=int(cfg.data.vocab_size),
                is_absorb=bool(self.proc.is_absorb),
                mask_id=(
                    int(self.proc.mask_id) if self.proc.is_absorb else None
                ),
                seq_len=seq_len,
                num_steps=int(
                    getattr(
                        getattr(cfg, "evaluation", object()),
                        "num_sampling_steps",
                        128,
                    )
                ),
                t_eps=float(getattr(cfg.diffusion.discrete, "eps", 1e-3)),
            )

            self.callbacks = []

            mauve_cfg = getattr(cfg.train, "mauve", None)
            if mauve_cfg is not None and bool(
                getattr(mauve_cfg, "enabled", False)
            ):
                self.callbacks.append(MauveCallback(cfg))

            vis_cfg = getattr(cfg.train, "visualization", None)
            if vis_cfg is not None and bool(
                getattr(vis_cfg, "enabled", False)
            ):
                self.callbacks.append(VisualizationCallback(cfg))

        else:
            raise ValueError(f"Unknown framework: {cfg.framework}")

        # ── resume ──────────────────────────────────────────────────────────
        self.global_step = 0
        self.resume_mode = "scratch"  # one of: scratch | init_from | resume
        self.start_epoch = self._resume()

        # Multimodal warm start (sequence-checkpoint column surgery) +
        # curriculum trunk freeze. Warm start only applies to a fresh
        # run; a true resume already carries the trained (and possibly
        # unfrozen) weights and its own freeze bookkeeping is re-derived
        # from the resumed global_step.
        if self.is_multimodal:
            if self.resume_mode == "scratch":
                self._mm_warm_start()
            self._mm_apply_initial_freeze()

        if self.is_master and self.tb is not None:
            self.tb.prepare_for_run(self.resume_mode)
            self.writer = self.tb.writer

        if self.ckpt_interval_enabled and self.ckpt_interval_every_steps > 0:
            gs = int(self.global_step)
            k = (gs // self.ckpt_interval_every_steps) + 1
            self._next_interval_ckpt_step = k * self.ckpt_interval_every_steps
        else:
            self._next_interval_ckpt_step = None

        if (
            self.resume_interval_enabled
            and self.resume_interval_every_steps > 0
        ):
            gs = int(self.global_step)
            k = (gs // self.resume_interval_every_steps) + 1
            self._next_resume_ckpt_step = k * self.resume_interval_every_steps
        else:
            self._next_resume_ckpt_step = None

        if self.cfg.framework == "continuous_score" and not getattr(
            self, "_entropy_restored_from_checkpoint", False
        ):
            self._load_entropy_tables_if_any()
            if (
                getattr(self, "_entropy_ready", False)
                and self.entropy_profile_source == "none"
            ):
                self.entropy_profile_source = "disk"

        if self.is_master:
            self._print_model_summary()

    # ──────────────────────────────────────────────────────────────────────
    # EMA and logging helpers
    # ──────────────────────────────────────────────────────────────────────
    def _update_ema(self) -> None:
        if self.ema_ramp_enabled:
            self.ema.decay = _ema_decay_for_step(
                self.ema_ramp_start,
                self.ema_ramp_end,
                self.ema_ramp_steps,
                self.global_step + 1,
            )
        self.ema.update(self.model)

    def _log_wandb(self, data: dict):
        if not self.use_wandb or not self.is_master:
            return
        # When TensorBoard sync is on, the same metrics already flow to
        # W&B via the mirrored TB writes (with the TB global_step as the
        # x-axis). Logging them again here would duplicate the series
        # and fight over the W&B step, so stand down and let the mirror
        # be the single source of truth.
        if getattr(self, "_wandb_sync_tb", False):
            return
        payload = dict(data)
        payload["global_step"] = int(self.global_step)
        wandb.log(payload)

    # ──────────────────────────────────────────────────────────────────────
    # Model summary
    # ──────────────────────────────────────────────────────────────────────
    def _print_model_summary(self):
        model_for_summary = _unwrap_all(self.model)
        total, trainable, non_trainable = _count_params(model_for_summary)
        first_param = next(model_for_summary.parameters(), None)
        p_dtype = (
            first_param.dtype if first_param is not None else torch.float32
        )
        amp_on = bool(self.cfg.train.use_fp16)
        amp_dtype = (
            "bf16"
            if (amp_on and torch.cuda.is_bf16_supported())
            else ("fp16" if amp_on else "fp32")
        )

        sampler_name = self.sampler.__class__.__name__
        sampler_steps = getattr(
            self.cfg.evaluation, "num_sampling_steps", None
        )

        print("\n" + "─" * 80)
        print(f"Experiment: {self.cfg.experiment}")
        print(
            f"Framework : {self.cfg.framework}  |  Device: {self.device} "
            f"(Rank {self.rank}/{self.world_size}) |  AMP: {amp_on} ("
            f"{amp_dtype})"
        )
        print(f"Model     : {model_for_summary.__class__.__name__}")
        print(
            f"Params    : total={_fmt_num(total)} ("
            f"{_human_bytes(total, p_dtype)}), trainable="
            f"{_fmt_num(trainable)}, frozen={_fmt_num(non_trainable)}"
        )
        print(
            f"Optimizer : {self.opt.__class__.__name__}  |  Scheduler: "
            f"{self.lr_sched.__class__.__name__}"
        )
        print(f"EMA       : decay={self.cfg.train.ema_decay}")
        print(
            f"Data      : dataset={self.cfg.data.dataset}, batch_size="
            f"{self.cfg.train.batch_size} (Global)"
        )
        if hasattr(self.cfg.data, "vocab_size"):
            extra = f"vocab_size={self.cfg.data.vocab_size}"
            if hasattr(self.cfg.data, "sequence_len"):
                extra += f", seq_len={self.cfg.data.sequence_len}"
            print(f"Discrete  : {extra}")
        if self.cfg.framework == "discrete_sedd":
            print(
                f"Diffusion : Q="
                f"{self.cfg.diffusion.discrete.q_matrix_type}, schedule="
                f"{self.cfg.diffusion.discrete.schedule}, t_max="
                f"{self.cfg.diffusion.discrete.t_max}"
            )
        print(
            f"Sampler   : {sampler_name}"
            + (
                f" (steps={sampler_steps})"
                if sampler_steps is not None
                else ""
            )
        )
        print(
            f"Resume    : start_epoch={self.start_epoch}, total_epochs="
            f"{self.cfg.train.epochs}, global_step={self.global_step}"
        )
        print(
            f"Paths     : run_dir={self.run_dir}, ckpt="
            f"{self._checkpoint_path()}"
        )
        print(
            f"Checkpointing: save_last={self.save_last}, save_top_k="
            f"{self.save_top_k}, mode={self.checkpoint_mode}"
        )
        self._write_artifact_manifest()
        print("─" * 80 + "\n")

    def _write_artifact_manifest(self) -> None:
        """State every output location once, in the log and as JSON."""
        if not self.is_master:
            return
        ev = getattr(self.cfg, "evaluation", None)
        jobname = os.environ.get("SLURM_JOB_NAME", "")
        jobid = os.environ.get("SLURM_JOB_ID", "")
        artifacts = {
            "experiment": str(getattr(self.cfg, "experiment", "")),
            "run_dir": str(self.run_dir),
            "checkpoints": {
                "resume": str(self._checkpoint_path("last")),
                "best": str(self._checkpoint_path("best")),
                "best_metadata": str(self.ckpt_dir / "best.json"),
                "progress": str(self.ckpt_dir / "progress.json"),
                "realized_mixture": str(
                    self.ckpt_dir / "realized_mixture.json"
                ),
            },
            "logs": {
                "tensorboard": str(self.run_dir / "training_logs"),
                "selection_run": str(
                    self.run_dir / "logs" / "checkpoint_selection.jsonl"
                ),
                "selection_index": "logs/selection/checkpoint_selection.jsonl",
                "slurm": (
                    f"logs/slurm/{jobname}-{jobid}.out"
                    if jobid
                    else "logs/slurm/"
                ),
            },
            "wandb": {
                "enabled": bool(self.use_wandb),
                "mode": os.environ.get("WANDB_MODE", "online"),
                "project": os.environ.get("WANDB_PROJECT", ""),
                "offline_dir": str(self.run_dir / "wandb"),
                "identity": str(self.run_dir / "wandb_run.json"),
                "sync": "scripts/launch/nogres/wandb_sync.sh  (login node)",
            },
            "evaluation": {
                "out_dir": str(getattr(ev, "out_dir", "")) if ev else "",
                "samples_dir": str(getattr(ev, "samples_dir", ""))
                if ev
                else "",
                "checkpoint_path": str(getattr(ev, "checkpoint_path", ""))
                if ev
                else "",
                "best_checkpoint_path": (
                    str(getattr(ev, "best_checkpoint_path", "")) if ev else ""
                ),
            },
        }
        print("Artifacts :")
        print(f"  checkpoints   {self.ckpt_dir}/")
        print(f"  tensorboard   {self.run_dir / 'training_logs'}/")
        print(
            f"  selection log {self.run_dir / 'logs'}"
            f"/checkpoint_selection.jsonl"
        )
        print(
            f"  wandb         {('on' if self.use_wandb else 'OFF')} mode="
            f"{artifacts['wandb']['mode']} project="
            f"{artifacts['wandb']['project'] or '(config default)'}"
        )
        if not self.use_wandb:
            print(
                "                W&B is off. Set PROTEBIT_WANDB=1 and unset "
                "WANDB_MODE=disabled to record this run."
            )
        try:
            (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
            with (self.run_dir / "artifacts.json").open("w") as handle:
                json.dump(artifacts, handle, indent=2)
        except OSError:
            pass

    def _maybe_run_sanity(self):
        s = getattr(self.cfg.train, "sanity", None)
        if s is None or (not bool(getattr(s, "enabled", False))):
            return

        sanity_epoch = int(getattr(s, "run_epoch", -1))

        if self.is_master:
            print(
                f"[sanity] running pre-train callbacks once (epoch="
                f"{sanity_epoch}) using ACTUAL config values..."
            )

        # Run the same hooks your training uses, once, before the epoch
        # loop. This will execute ALL-RANK callbacks safely if they set
        # run_on_all_ranks=True.
        self._run_callbacks("on_epoch_end", sanity_epoch)
        self._run_callbacks("on_train_epoch_end", sanity_epoch)

        if self.ddp_active:
            dist.barrier()

        if self.is_master:
            print("[sanity] done.")

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing / Resume
    # ──────────────────────────────────────────────────────────────────────
    def _checkpoint_path(self, name: str = "last") -> Path:
        return self.ckpt_dir / f"{name}.pt"

    def _rng_state(self):
        state = {
            "py_random": random.getstate(),
            "np_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _set_rng_state(self, state: dict):
        try:
            random.setstate(state["py_random"])
            np.random.set_state(state["np_random"])
            torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as e:
            if self.is_master:
                print(f"⚠️  RNG state restore warning: {e}")

    def _collect_rng_states(self):
        """Collect one stochastic state per rank for an exact DDP resume."""
        local = self._rng_state()
        if not self.ddp_active:
            return [local]
        states = [None for _ in range(self.world_size)]
        dist.all_gather_object(states, local)
        return states

    def _entropy_state(self) -> dict:
        """Return all online entropy state needed for an exact continuation."""

        def cpu(value):
            return (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )

        return {
            "schema_version": 1,
            "signature": {
                "buffer_size": int(self.entropy_buffer_size),
                "num_bins": int(self.entropy_num_bins),
                "warmup_steps": int(self.entropy_warmup_steps),
                "transition_steps": int(self.entropy_transition_steps),
                "gamma_max": float(self.entropy_gamma_max),
                "mode": str(self.entropy_mode),
                "regularizer_c": float(self.entropy_regularizer_c),
                "regularizer_n": float(self.entropy_regularizer_n),
                "rate_power": float(self.entropy_rate_power),
            },
            "sig_buffer": cpu(self._entropy_sig_buf),
            "metric_buffer": cpu(self._entropy_metric_buf),
            "buffer_ptr": int(self._entropy_buf_ptr),
            "buffer_len": int(self._entropy_buf_len),
            "ready": bool(self._entropy_ready),
            "pdf": cpu(self._entropy_pdf),
            "cdf": cpu(self._entropy_cdf),
            "sigmas": cpu(self._entropy_sigmas),
            "edges": cpu(self._entropy_edges),
            "ln_mu": self._entropy_ln_mu,
            "ln_std": self._entropy_ln_std,
            "profile_source": str(self.entropy_profile_source),
        }

    def _raise_distributed_checkpoint_error(self, error, kind: str) -> None:
        """
        Broadcast a rank-0 write result so no worker runs past a failed
        save.
        """
        if self.ddp_active:
            status = [error if self.is_master else None]
            dist.broadcast_object_list(status, src=0)
            error = status[0]
        if error:
            raise RuntimeError(
                f"{kind} checkpoint write failed on rank 0: {error}"
            )

    def _restore_entropy_state(self, state: dict) -> None:
        expected = self._entropy_state()["signature"]
        if dict(state.get("signature", {})) != expected:
            raise RuntimeError(
                "checkpoint entropy-controller configuration differs from the "
                "current run; exact resume is impossible"
            )
        sig = state.get("sig_buffer")
        metric = state.get("metric_buffer")
        if not isinstance(sig, torch.Tensor) or not isinstance(
            metric, torch.Tensor
        ):
            raise RuntimeError(
                "checkpoint entropy state is missing FIFO tensors"
            )
        if (
            sig.numel() != self.entropy_buffer_size
            or metric.numel() != self.entropy_buffer_size
        ):
            raise RuntimeError(
                "checkpoint entropy FIFO size does not match the configuration"
            )
        self._entropy_sig_buf.copy_(sig.to("cpu", dtype=torch.float32))
        self._entropy_metric_buf.copy_(metric.to("cpu", dtype=torch.float32))
        self._entropy_buf_ptr = int(state["buffer_ptr"])
        self._entropy_buf_len = int(state["buffer_len"])
        self._entropy_ready = bool(state.get("ready", False))
        for key, attr in (
            ("pdf", "_entropy_pdf"),
            ("cdf", "_entropy_cdf"),
            ("sigmas", "_entropy_sigmas"),
            ("edges", "_entropy_edges"),
        ):
            value = state.get(key)
            setattr(
                self,
                attr,
                value.to(self.device)
                if isinstance(value, torch.Tensor)
                else None,
            )
        self._entropy_ln_mu = state.get("ln_mu")
        self._entropy_ln_std = state.get("ln_std")
        self.entropy_profile_source = str(
            state.get("profile_source", self.entropy_profile_source)
        )
        self._entropy_restored_from_checkpoint = True

    def _load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if self._protebit_checkpoint_provenance is not None:
            saved_provenance = ckpt.get("protebit_provenance")
            if saved_provenance != self._protebit_checkpoint_provenance:
                raise RuntimeError(
                    "checkpoint provenance differs from this immutable "
                    "launch; refusing cross-attempt or cross-recipe "
                    "resume"
                )
        state_dict = ckpt["model"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("module."):
                k = k[7:]
            clean_state_dict[k] = v

        model_to_load = _unwrap_all(self.model)

        # A partial load must never be silent.
        #
        # `strict=False` on a state-dict mismatch is the single most
        # dangerous line in a chained run: link 1 trains a model,
        # someone edits a shape key, and links 2..94 each load whatever
        # subset still matches and carry on. The run completes, the loss
        # looks plausible because most of the trunk did load, and every
        # capacity claim about it is void. Nothing in the log said so.
        #
        # The fallback is still allowed -- adding a genuinely new module
        # (the P0 release added six) is a legitimate reason to load
        # non-strictly -- but it now has to be asked for, and it reports
        # exactly what it did.
        try:
            model_to_load.load_state_dict(clean_state_dict, strict=True)
        except RuntimeError as exc:
            incompatible = model_to_load.load_state_dict(
                clean_state_dict, strict=False
            )
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            n_params = sum(1 for _ in model_to_load.parameters())
            allow = bool(getattr(self.cfg.train, "allow_partial_load", False))
            detail = (
                f"checkpoint {path} does not match the model.\n  missing "
                f"from checkpoint ({len(missing)} of ~{n_params} tensors):"
                f" {missing[:8]}{(' ...' if len(missing) > 8 else '')}\n  "
                f"unexpected in checkpoint ({len(unexpected)}): "
                f"{unexpected[:8]}"
                f"{(' ...' if len(unexpected) > 8 else '')}\n  torch said:"
                f" {str(exc).splitlines()[0][:200]}"
            )
            if not allow:
                raise RuntimeError(
                    detail
                    + "\n\nRefusing a silent partial load. Either the model "
                    "shape changed since this checkpoint was written (start a "
                    "new run, or fix the config), or you are deliberately "
                    "warm-starting across a shape change -- in which case set "
                    "cfg.train.allow_partial_load = True."
                ) from exc
            if self.is_master:
                print(
                    "[resume] PARTIAL LOAD (allow_partial_load=True)\n"
                    + detail
                )

        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if "lr_sched" in ckpt:
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
        if (
            self.use_scaler
            and ("scaler" in ckpt)
            and (ckpt["scaler"] is not None)
        ):
            self.scaler.load_state_dict(ckpt["scaler"])

        if "ema" in ckpt and ckpt["ema"] is not None:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.to(self.device)

        rank_states = ckpt.get("rng_states_by_rank")
        if rank_states is not None:
            if len(rank_states) != self.world_size:
                raise RuntimeError(
                    f"checkpoint RNG world size mismatch: saved "
                    f"{len(rank_states)}, current {self.world_size}"
                )
            self._set_rng_state(rank_states[self.rank])
            if self.is_master:
                print("[resume] restored exact rank-specific RNG streams")
        elif "rng_state" in ckpt and ckpt["rng_state"] is not None:
            if self.world_size != 1:
                raise RuntimeError(
                    "legacy checkpoint contains only rank-0 RNG state and "
                    "cannot be called an exact distributed resume"
                )
            self._set_rng_state(ckpt["rng_state"])

        entropy_state = ckpt.get("entropy_state")
        if entropy_state is not None:
            # Schedule recomputation is rank-0 authoritative and
            # broadcast to workers, so one complete rank-0 FIFO/table
            # snapshot is both exact and avoids all-gathering ~200 MB of
            # redundant buffers at 32 ranks.
            self._restore_entropy_state(entropy_state)
        elif self.entropy_compute or self.entropy_use_for_sampling:
            raise RuntimeError(
                "checkpoint has no online entropy-controller state; "
                "exact resume for an entropic training run is impossible"
            )

        self.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.best_ckpts = ckpt.get("best_ckpts", self.best_ckpts)
        self.early_stop_best = ckpt.get("early_stop_best", self.best_metric)
        self.early_stop_bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0))

        # Multimodal exact-resume: restore the batch-sampler cursor and
        # collator RNG/counter, and if the checkpoint was taken
        # mid-epoch, re-enter that same epoch at the saved batch cursor
        # instead of skipping to the next epoch (gate 5). Non-multimodal
        # resume is unchanged.
        self._mm_resume_cursor = 0
        self._mm_resume_epoch = -1
        if self.is_multimodal:
            bs = getattr(self.train_loader, "batch_sampler", None)
            coll = getattr(self.train_loader, "collate_fn", None)
            if (
                "mm_sampler" in ckpt
                and bs is not None
                and hasattr(bs, "load_state_dict")
            ):
                bs.load_state_dict(ckpt["mm_sampler"])
            if (
                "mm_collator" in ckpt
                and coll is not None
                and hasattr(coll, "load_state_dict")
            ):
                coll.load_state_dict(ckpt["mm_collator"])
            self._mm_mix_counts = {
                str(k): int(v)
                for k, v in dict(ckpt.get("mm_mix_counts", {})).items()
            }
            self._mm_mix_total = int(ckpt.get("mm_mix_total", 0))
            self._mm_corpus_microbatches = int(
                ckpt.get("mm_corpus_microbatches", 0)
            )
            self._mm_replay_microbatches = int(
                ckpt.get("mm_replay_microbatches", 0)
            )
            self._mm_active_phase = int(ckpt.get("mm_active_phase", -1))
            self._replay_batches_drawn_epoch = int(
                ckpt.get("mm_replay_batches_drawn_epoch", 0)
            )
            self._mm_resume_replay_cursor = self._replay_batches_drawn_epoch
            cursor = int(ckpt.get("mm_batch_cursor", 0))
            steps_per_epoch = int(
                getattr(self.cfg.train, "steps_per_epoch", 0)
            )
            # Only re-enter the same epoch when the batch sampler can
            # actually seek to a cursor (SourceLengthBatchSampler). A
            # non-seekable fallback sampler (e.g.
            # DistributedLengthBucketBatchSampler when
            # steps_per_epoch<=0) would restart the epoch from batch 0,
            # so fall back to the standard next-epoch resume instead.
            sampler_seekable = bs is not None and hasattr(bs, "start_batch")
            # `cursor` is `_epoch_batches_done`, counted in
            # MICRO-batches; `steps_per_epoch` counts OPTIMIZER steps.
            # The two were only accidentally comparable while the loader
            # was given num_batches=steps_per_epoch. Now that an epoch
            # is steps_per_epoch * grad_accum micro-batches, comparing
            # against the optimizer-step count would declare every
            # preemption past the halfway point "past the end of the
            # epoch", skip to epoch+1, and silently discard the
            # remaining optimizer steps -- on roughly half of all
            # 4-hour-link preemptions.
            _accum_resume = max(
                1, int(getattr(self.cfg.train, "grad_accum_steps", 1) or 1)
            )
            micro_per_epoch = steps_per_epoch * _accum_resume
            if (
                cursor > 0
                and sampler_seekable
                and (steps_per_epoch <= 0 or cursor < micro_per_epoch)
            ):
                # Rolling checkpoint mid-epoch: continue the same epoch.
                self._mm_resume_cursor = cursor
                self._mm_resume_epoch = int(ckpt.get("epoch", -1))
                start_epoch = int(ckpt.get("epoch", -1))

        if self.is_master:
            print(f"Resumed from {path} (Epoch {start_epoch})")
        return start_epoch

    def _clean_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Make checkpoint state_dict portable across:
          - torch.compile (_orig_mod.)
          - DDP (module.)
        """
        clean = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("module."):
                k = k[7:]
            clean[k] = v
        return clean

    def _init_from_checkpoint_weights_only(self, path: Path):
        """
        Initialize:
          - running model weights   <- ckpt["model"]
          - EMA shadow weights      <- ckpt["ema"] (if present)
        WITHOUT resuming optimizer/scheduler/scaler/global_step/epoch.

        This is the "fresh run from weights" mode.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model" not in ckpt:
            raise KeyError(f"Checkpoint at {path} missing key 'model'.")

        # --- load running weights into the running model ---
        model_sd = self._clean_state_dict_keys(ckpt["model"])
        model_to_load = _unwrap_all(self.model)
        try:
            model_to_load.load_state_dict(model_sd, strict=True)
        except RuntimeError:
            model_to_load.load_state_dict(model_sd, strict=False)

        # --- load EMA weights into EMA object (shadow) if present &
        # enabled ---
        use_ema = bool(getattr(self.cfg.train, "init_from_use_ema", True))
        if use_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
            try:
                # Your checkpoints store: {"decay": float, "shadow":
                # {name: tensor(cpu), ...}}
                self.ema.load_state_dict(ckpt["ema"])
                self.ema.to(self.device)
            except Exception as e:
                if self.is_master:
                    print(
                        f"⚠️  init_from: failed to load EMA state ("
                        f"{type(e).__name__}: {e}). Continuing with EMA "
                        f"re-initialized from running weights."
                    )
                # fallback: re-init EMA shadow from current model
                # weights
                self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
                self.ema.to(self.device)
        else:
            # If EMA load disabled/unavailable: re-init EMA from current
            # model weights.
            self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
            self.ema.to(self.device)

        # --- reset training bookkeeping (fresh run) ---
        self.global_step = 0
        self.best_metric = (
            math.inf if self.checkpoint_mode == "min" else -math.inf
        )
        self.best_ckpts = []
        self.early_stop_best = self.best_metric
        self.early_stop_bad_epochs = 0

        if self.is_master:
            print(f"[init_from] Initialized running weights from: {path}")
            if use_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
                print("[init_from] Loaded EMA shadow from checkpoint as well.")
            else:
                print(
                    "[init_from] EMA shadow initialized from running "
                    "weights (no EMA loaded)."
                )

    def _resume(self) -> int:
        ckpt_path = (
            self._checkpoint_path()
        )  # runs/<experiment>/checkpoints/last.pt

        init_from = getattr(self.cfg.train, "init_from", None)
        init_force = bool(getattr(self.cfg.train, "init_from_force", False))

        # `init_from: "best"` resolves to the previous stage's
        # task-selected best checkpoint. This is the level at which
        # "always relaunch from the best checkpoint" is the right rule:
        # starting a NEW stage (S0 -> S1, or a fine-tune) should begin
        # from the best weights, and every evaluation should score them.
        #
        # It is deliberately NOT the rule for an intra-chain resume.
        # best.pt is an earlier-step snapshot, so resuming a chain link
        # from it would throw away real progress and the LR-schedule
        # position, and it can livelock: link resumes from best, best
        # does not improve, the next link resumes from the same best,
        # and the progress guard cancels the chain because global_step
        # never advanced. last.pt keeps the chain exact; best.pt is what
        # leaves the run.
        if isinstance(init_from, str) and init_from.strip().lower() == "best":
            resolved = self._resolve_best_checkpoint()
            if resolved is None:
                raise FileNotFoundError(
                    "cfg.train.init_from='best' but no best checkpoint was "
                    "found. Set cfg.train.init_from_run to the run directory "
                    "holding checkpoints/best.pt."
                )
            init_from = str(resolved)
            if self.is_master:
                print(f"[init_from] resolved 'best' -> {init_from}")

        # Opt-in: weights-only initialization from a specified
        # checkpoint. This is NOT a true resume of training history.
        if init_from is not None:
            init_path = Path(init_from)
            if init_force or (not ckpt_path.exists()):
                if not init_path.exists():
                    raise FileNotFoundError(
                        f"cfg.train.init_from not found: {init_path}"
                    )
                if self.is_master:
                    print(
                        f"[init_from] force={init_force} | last_exists="
                        f"{ckpt_path.exists()} | path={init_path}"
                    )
                self._init_from_checkpoint_weights_only(init_path)
                self.resume_mode = "init_from"
                return 0

        # True training resume from last.pt
        if ckpt_path.exists():
            self.resume_mode = "resume"
            return self._load_checkpoint(ckpt_path)

        # Fresh run from scratch
        self.resume_mode = "scratch"
        if self.is_master:
            print("🏁 Starting training from scratch.")
        return 0

    def _resolve_best_checkpoint(self) -> Path | None:
        """Locate the best checkpoint of the run named by init_from_run."""
        run = getattr(self.cfg.train, "init_from_run", None)
        base = Path(run) if run else Path("runs") / str(self.cfg.experiment)
        ckpt_dir = base / "checkpoints"
        sidecar = ckpt_dir / "best.json"
        if sidecar.exists():
            try:
                info = json.loads(sidecar.read_text())
                metric = str(info.get("metric", "unknown"))
                if self.is_master and metric not in {"task_score"}:
                    print(
                        f"[init_from] WARNING: {sidecar} says best.pt was "
                        f"selected on {metric!r}, not 'task_score'. On "
                        f"this model, denoising-loss selection measurably "
                        f"cost 18 AAR points. Evaluate last.pt as well "
                        f"before trusting this warm start."
                    )
                candidate = ckpt_dir / str(info.get("path", "best.pt"))
                if candidate.exists():
                    return candidate
            except (OSError, ValueError):
                pass
        candidate = ckpt_dir / "best.pt"
        return candidate if candidate.exists() else None

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _run_callbacks(self, method_name: str, *args):
        for cb in self.callbacks:
            run_all = bool(getattr(cb, "run_on_all_ranks", False))
            if self.is_master or run_all:
                if hasattr(cb, method_name):
                    if self.is_master and method_name == "on_epoch_end":
                        print(
                            f"[callback] before {cb.__class__.__name__}: "
                            f"{_gpu_mem_msg(self.device)}"
                        )
                    getattr(cb, method_name)(self, *args)
                    if self.is_master and method_name == "on_epoch_end":
                        print(
                            f"[callback] after  {cb.__class__.__name__}: "
                            f"{_gpu_mem_msg(self.device)}"
                        )

    # ──────────────────────────────────────────────────────────────────────
    # Multimodal (18-bit paired) training path: dict batches, warm
    # start, trunk freeze/unfreeze, lower trunk LR, and sequence-only
    # replay (gates 2/3). Every method below is inert unless
    # cfg.data.dataset selects the paired multimodal corpus, so the
    # sequence-only paths are byte-for-byte unchanged.
    # ──────────────────────────────────────────────────────────────────────
    _MM_ADAPTER_PREFIXES = (
        "patch_proj",
        "unpatch_proj_content",
        "head",
        "mm_embed",
    )

    def _mm_is_trunk_param(self, name: str) -> bool:
        """
        True for warm-started shared params; False for new multimodal
        adapters.
        """
        n = name.replace("_orig_mod.", "")
        if n.startswith("module."):
            n = n[7:]
        top = n.split(".", 1)[0]
        return top not in self._MM_ADAPTER_PREFIXES

    def _mm_build_param_groups(self):
        raw = _unwrap_all(self.model)
        adapter, trunk = [], []
        for name, p in raw.named_parameters():
            (trunk if self._mm_is_trunk_param(name) else adapter).append(p)
        # Order matters: group 0 = adapters, group 1 = trunk (scaled
        # each step).
        return [
            {"params": adapter, "mm_group": "adapter"},
            {"params": trunk, "mm_group": "trunk"},
        ]

    def _mm_set_trunk_requires_grad(self, flag: bool) -> None:
        raw = _unwrap_all(self.model)
        for name, p in raw.named_parameters():
            if self._mm_is_trunk_param(name):
                p.requires_grad_(bool(flag))

    def _mm_apply_initial_freeze(self) -> None:
        """Freeze the trunk iff still inside the warm-start freeze window."""
        if self.freeze_trunk_steps <= 0:
            self._trunk_frozen = False
            return
        should_freeze = int(self.global_step) < int(self.freeze_trunk_steps)
        self._mm_set_trunk_requires_grad(not should_freeze)
        self._trunk_frozen = should_freeze
        if should_freeze and self.is_master:
            print(
                f"[multimodal] trunk frozen for warm start until step "
                f"{self.freeze_trunk_steps} (adapters train first)."
            )

    def _mm_maybe_unfreeze(self) -> None:
        if self._trunk_frozen and int(self.global_step) >= int(
            self.freeze_trunk_steps
        ):
            self._mm_set_trunk_requires_grad(True)
            self._trunk_frozen = False
            if self.is_master:
                print(
                    f"[multimodal] unfroze trunk at step "
                    f"{self.global_step} (trunk_lr_mult="
                    f"{self.trunk_lr_mult})."
                )

    def _mm_warm_start(self) -> None:
        """
        Column-surgery warm start from a sequence-only checkpoint (plan
        6.4).
        """
        path = self.warm_start_seq_checkpoint
        if not path:
            return
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"cfg.model.warm_start_seq_checkpoint not found: {p}"
            )
        from utils.protein_warmstart import warm_start_multimodal_from_sequence

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        seq_sd = None
        if isinstance(ckpt, dict):
            use_ema = bool(getattr(self.cfg.model, "warm_start_use_ema", True))
            ema_obj = ckpt.get("ema") if use_ema else None
            if isinstance(ema_obj, dict) and "shadow" in ema_obj:
                seq_sd = ema_obj["shadow"]
            if seq_sd is None:
                seq_sd = ckpt.get("model", ckpt)
        else:
            seq_sd = ckpt
        report = warm_start_multimodal_from_sequence(
            _unwrap_all(self.model), seq_sd, verbose=self.is_master
        )
        # Acceptance criterion by PARAMETER fraction, not "at least one
        # tensor".
        #
        # The old guard fired only when nothing at all was copied, so a
        # transfer of exactly ONE scalar out of 126,497,953 passed it --
        # measured, on the 384-wide EvoDiff checkpoint against the
        # 640-wide model. The run then printed a warm-start report that
        # read like success, froze 120M randomly-initialised parameters
        # for 2,000 steps to protect them, and paid DDP
        # find_unused_parameters for the entire run.
        #
        # A warm start that transfers a rounding error is not a warm
        # start; it is a random init with extra steps and a misleading
        # log line.
        transferred = int(report.get("transferred_params", 0))
        target = int(report.get("target_params", 0)) or 1
        fraction = transferred / target
        self._warm_start_fraction = fraction
        floor = float(getattr(self.cfg.model, "warm_start_min_fraction", 0.5))
        if fraction < floor:
            allow = bool(
                getattr(self.cfg.model, "warm_start_allow_partial", False)
            )
            msg = (
                f"warm start from {p} transferred {transferred:,} of "
                f"{target:,} parameters ({fraction:.2%}), below the "
                f"warm_start_min_fraction floor of {floor:.0%}.\n  report:"
                f" {report}\n  This usually means the source and target "
                f"trunks differ in width or depth. No width-matched "
                f"sequence checkpoint exists until S0 has run "
                f"(configs/proteins/s0_sequence_pretrain_400m.py); until "
                f"then set cfg.model.warm_start_seq_checkpoint = '' and "
                f"train from scratch.\n  Set "
                f"cfg.model.warm_start_allow_partial = True to proceed "
                f"anyway, which is only sensible for a deliberate partial "
                f"transfer."
            )
            if not allow:
                raise RuntimeError(msg)
            if self.is_master:
                print("[warm_start] WARNING " + msg)
        # Re-seed the EMA shadow from the warm-started weights so
        # evaluation does not average toward the discarded random init.
        self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
        self.ema.to(self.device)
        if fraction < floor and self.freeze_trunk_steps > 0:
            self.freeze_trunk_steps = 0
            if self.is_master:
                print(
                    "[warm_start] trunk freeze disabled: the transfer did not "
                    "clear the floor, so there is nothing to protect."
                )
        if self.is_master:
            print(
                f"[multimodal] warm start from {p}: {report} -> "
                f"{fraction:.1%} of parameters"
            )

    def _mm_next_replay_batch(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        try:
            batch = next(self._replay_iter)
        except StopIteration:
            self._replay_iter = iter(self.replay_loader)
            batch = next(self._replay_iter)
        self._replay_batches_drawn_epoch += 1
        return batch

    def _mm_progress(self) -> float:
        """Training progress in [0, 1] for the mixture schedule."""
        total = int(getattr(self.cfg.optim, "total_steps", 0) or 0)
        if total <= 0:
            return 0.0
        return min(1.0, max(0.0, float(self.global_step) / float(total)))

    def _mm_should_replay(self) -> bool:
        """Deterministic per-MICRO-STEP corpus draw (resumable)."""
        if not self.is_multimodal or self.replay_loader is None:
            return False
        fraction = self._mm_replay_fraction_now()
        if fraction <= 0.0:
            return False
        seed = int(getattr(self.cfg.train, "seed", 42))
        if self.replay_exact_window:
            quota = self.replay_exact_count / self.replay_exact_window
            if not math.isclose(fraction, quota, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"scheduled replay fraction does not match its exact "
                    f"quota: {fraction} != {self.replay_exact_count}/"
                    f"{self.replay_exact_window}"
                )
            accum = max(1, int(getattr(self.cfg.train, "grad_accum_steps", 1)))
            micro_step = int(self.global_step) * accum + int(
                self._accum_counter
            )
            block, position = divmod(micro_step, self.replay_exact_window)
            # Shuffle the occupied slots independently in every quota
            # block so replay is not coupled to a fixed optimizer-step
            # phase. The keyed draw makes resume reproduce the identical
            # choice without storing a mutable RNG stream.
            occupied = np.random.default_rng((seed, 778, block)).choice(
                self.replay_exact_window,
                size=self.replay_exact_count,
                replace=False,
            )
            return bool(position in occupied)
        draw = np.random.default_rng(
            (seed, 777, int(self.global_step), int(self._epoch_batches_done))
        ).random()
        return bool(draw < fraction)

    def _mm_replay_fraction_now(self) -> float:
        """Replay share of draws at the current progress."""
        if self.mixture_schedule is not None:
            from data.protein_tasks import REPLAY_CORPUS

            return float(
                self.mixture_schedule.corpus_fraction(
                    self._mm_progress(), REPLAY_CORPUS
                )
            )
        return float(self.replay_fraction)

    def _mm_refresh_mixture(self, force: bool = False) -> None:
        """Push the phase's task weights into both collators."""
        if self.mixture_schedule is None:
            return
        from data.protein_tasks import PAIRED_CORPUS, REPLAY_CORPUS

        progress = self._mm_progress()
        phase = self.mixture_schedule.phase_index(progress)
        if (not force) and phase == self._mm_active_phase:
            return
        self._mm_active_phase = phase
        paired = self.mixture_schedule.task_weights_for_corpus(
            progress, PAIRED_CORPUS
        )
        replay = self.mixture_schedule.task_weights_for_corpus(
            progress, REPLAY_CORPUS
        )
        for loader, weights in (
            (self.train_loader, paired),
            (self.replay_loader, replay),
        ):
            if loader is None or not weights:
                continue
            collate = getattr(loader, "collate_fn", None)
            if collate is not None and hasattr(collate, "task_weights"):
                collate.task_weights = dict(weights)
        if self.is_master:
            print(
                f"[mixture] phase {phase} at progress {progress:.3f}: "
                f"replay={self._mm_replay_fraction_now():.3f} paired="
                f"{ {k: round(v, 3) for (k, v) in paired.items()} }"
            )

    def _mm_record_mixture(self, batch, did_replay: bool) -> None:
        """Count the REALIZED (corpus, task) proportions."""
        from data.protein_tasks import PAIRED_CORPUS, REPLAY_CORPUS

        corpus = REPLAY_CORPUS if did_replay else PAIRED_CORPUS
        self._mm_corpus_microbatches += 1
        if did_replay:
            self._mm_replay_microbatches += 1
        tasks = None
        if isinstance(batch, dict):
            tasks = batch.get("task_names") or batch.get("tasks")
        if not tasks:
            return
        for task in tasks:
            key = f"{corpus}/{task}"
            self._mm_mix_counts[key] = self._mm_mix_counts.get(key, 0) + 1
            self._mm_mix_total += 1

    def _mm_realized_mixture(self) -> dict[str, float]:
        """Realized draw proportions so far, as {'corpus/task': fraction}."""
        if self._mm_mix_total <= 0:
            return {}
        total = float(self._mm_mix_total)
        return {k: v / total for k, v in sorted(self._mm_mix_counts.items())}

    def _mm_log_mixture(self, every: int = 1000) -> None:
        """Emit train/mix/<corpus>/<task> scalars on a step cadence."""
        if not self.is_master or self.mixture_schedule is None:
            return
        if self._mm_mix_total <= 0:
            return
        if (self.global_step % max(1, int(every))) != 0:
            return
        if self.global_step == self._mm_mix_logged_step:
            return
        self._mm_mix_logged_step = int(self.global_step)
        realized = self._mm_realized_mixture()
        payload = {f"train/mix/{k}": v for k, v in realized.items()}
        for key, value in payload.items():
            self.writer.add_scalar(key, value, self.global_step)
        self._log_wandb(payload)

    def _mm_assert_mixture(self, tolerance: float = 0.02) -> None:
        """
        Compare realized against configured at run end and report the
        gap.
        """
        if not self.is_master or self.mixture_schedule is None:
            return
        realized = self._mm_realized_mixture()
        if not realized:
            return
        exact_replay = None
        if self.replay_exact_window and self._mm_corpus_microbatches:
            exact_replay = {
                "replay_microbatches": int(self._mm_replay_microbatches),
                "total_microbatches": int(self._mm_corpus_microbatches),
                "realized_fraction": float(
                    self._mm_replay_microbatches / self._mm_corpus_microbatches
                ),
                "quota_count": int(self.replay_exact_count),
                "quota_window": int(self.replay_exact_window),
            }
        configured = {
            f"{c}/{t}": w
            for (c, t), w in self.mixture_schedule.weights_at(
                self._mm_progress()
            ).items()
        }
        rows = []
        worst = 0.0
        for key in sorted(set(realized) | set(configured)):
            got = realized.get(key, 0.0)
            want = configured.get(key, 0.0)
            worst = max(worst, abs(got - want))
            rows.append({"key": key, "realized": got, "configured": want})
        verdict = "OK" if worst <= tolerance else "FAIL"
        print(
            f"[mixture] realized vs configured over {self._mm_mix_total:,}"
            f" examples: worst gap {worst:.4f} (tolerance {tolerance}) -> "
            f"{verdict}"
        )
        if verdict == "FAIL":
            for row in rows:
                if abs(row["realized"] - row["configured"]) > tolerance:
                    print(
                        f"  {row['key']:<40s} realized "
                        f"{row['realized']:.4f} configured "
                        f"{row['configured']:.4f}"
                    )
            print(
                "  The realized mixture is what the optimizer saw. Note that "
                "the final phase's weights are compared against counts "
                "accumulated over ALL phases, so a large gap is expected on a "
                "run that changed phase; read the per-phase W&B series."
            )
        try:
            with (self.ckpt_dir / "realized_mixture.json").open("w") as handle:
                json.dump(
                    {
                        "examples": int(self._mm_mix_total),
                        "worst_gap": float(worst),
                        "tolerance": float(tolerance),
                        "verdict": verdict,
                        "rows": rows,
                        "exact_replay_quota": exact_replay,
                    },
                    handle,
                    indent=2,
                )
        except OSError:
            pass

    def _step_multimodal(self, batch, is_train: bool):
        """One 18-bit paired micro-step with gradient accumulation."""
        from trainers.multimodal_step import multimodal_training_step

        if is_train:
            # Unfreeze before the forward so the boundary step trains
            # the trunk.
            self._mm_maybe_unfreeze()

        accum = max(1, int(getattr(self, "grad_accum_steps", 1)))
        is_boundary = (not is_train) or ((self._accum_counter + 1) >= accum)
        # Suppress DDP gradient sync on the non-final micro-steps of a
        # window.
        suppress_sync = (
            is_train
            and accum > 1
            and not is_boundary
            and self.ddp_active
            and hasattr(self.model, "no_sync")
        )
        sync_ctx = (
            self.model.no_sync() if suppress_sync else contextlib.nullcontext()
        )

        # Online entropy schedule for the paired path. draw_sigma()
        # returns the log-normal/EDM base until the schedule is warmed
        # up + ready, then blends toward the entropic schedule (gamma
        # ramp), exactly like
        # _step_continuous.
        # Only engage on training micro-steps and only when entropy is
        # configured, so every other multimodal config is byte-for-byte
        # unchanged.
        entropy_on = (
            is_train
            and self.cfg.framework == "continuous_score"
            and (self.entropy_compute or self.entropy_use_for_sampling)
        )
        sigma_draw_fn = self.entropy_ctrl.draw_sigma if entropy_on else None
        entropy_sink = {} if (is_train and self.entropy_compute) else None
        diagnostics_sink = {} if not is_train else None

        amp_enabled = bool(self.cfg.train.use_fp16)
        with autocast(
            self.device.type, enabled=amp_enabled, dtype=self.amp_dtype
        ):
            loss, components = multimodal_training_step(
                self.model,
                batch,
                self.proc,
                self.cfg,
                device=self.device,
                is_train=is_train,
                sigma_draw_fn=sigma_draw_fn,
                entropy_sink=entropy_sink,
                diagnostics_sink=diagnostics_sink,
            )

        # Feed the entropy FIFO buffer with this micro-batch's valid
        # per-modality (sigma, denoising-MSE) pairs (both modalities,
        # independent sigmas). Runs every micro-batch under grad
        # accumulation, mirroring
        # _step_continuous.
        if entropy_sink:
            vs = entropy_sink["valid_seq"]
            vt = entropy_sink["valid_struct"]
            sig = torch.cat(
                [
                    entropy_sink["sigma_seq"][vs],
                    entropy_sink["sigma_struct"][vt],
                ]
            )
            met = torch.cat(
                [
                    entropy_sink["metric_seq"][vs],
                    entropy_sink["metric_struct"][vt],
                ]
            )
            if sig.numel() > 0:
                self._update_entropy_buffer(sig, met)

        if is_train:
            if self._accum_counter == 0:
                self.opt.zero_grad(set_to_none=True)
            scaled = loss / accum  # so accumulated grads average the window
            with sync_ctx:
                if self.use_scaler:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if is_boundary:
                if self.use_scaler:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.opt.step()

                self.lr_sched.step()
                # The scheduler forces one lr on every group; scale the
                # trunk group (group 1) back down so the warm-started
                # backbone moves slower.
                if self._trunk_group_active:
                    self.opt.param_groups[1]["lr"] *= self.trunk_lr_mult
                self._update_ema()
                self._accum_counter = 0
                self._did_optim_step = True
            else:
                self._accum_counter += 1
                self._did_optim_step = False
        else:
            self._did_optim_step = True

        self._last_mm_components = {k: float(v) for k, v in components.items()}
        if diagnostics_sink is not None:
            diagnostics_sink["task_names"] = list(batch.get("task_names", []))
            self._last_mm_val_diagnostics = diagnostics_sink
        else:
            self._last_mm_val_diagnostics = None
        return loss.item()

    # ------------------------------------------------------------------
    # Trainer method: full step_continuous
    # ------------------------------------------------------------------
    def _step_continuous(self, x0, is_train: bool):
        """
        Continuous score training step with optional CFG-style prefix
        conditioning.

        Supports:
        - binary continuous diffusion: x_t [B,S], logits [B,S] or
        [B,S,1]
        - one-hot token continuous diffusion: x_t [B,S,V], logits
        [B,S,V]
        - optional prefix conditioning
        - optional self-conditioning

        Memory-oriented implementation:
        - token one-hot state uses AMP dtype when enabled
        - xt is built in-place (no standalone noise tensor)
        - x0_hat allocated only if actually needed
        - large temporaries freed as early as possible
        """
        B = x0.size(0)
        repr_mode = str(
            getattr(self.cfg.data, "representation", "binary")
        ).lower()
        is_cont_tokens = repr_mode == "tokens"

        amp_enabled = bool(getattr(self.cfg.train, "use_fp16", False))
        state_dtype = self.amp_dtype if amp_enabled else torch.float32

        # ------------------------------------------------------------------
        # Prepare clean target / clean state
        # ------------------------------------------------------------------
        if is_cont_tokens:
            V = int(self.cfg.data.vocab_size)
            x0_target = (
                x0.to(self.device, non_blocking=True).long().view(B, -1)
            )  # [B,S]
            S = x0_target.size(1)

            # Dense one-hot clean state in reduced precision when AMP is
            # enabled.
            x0_clean = torch.nn.functional.one_hot(
                x0_target, num_classes=V
            ).to(dtype=state_dtype)  # [B,S,V]
        else:
            x0_clean = (
                x0.to(self.device, non_blocking=True)
                .view(B, -1)
                .to(dtype=torch.float32)
            )  # [B,S]
            S = x0_clean.size(1)
            x0_target = None

        # ------------------------------------------------------------------
        # Conditioning setup
        # ------------------------------------------------------------------
        cond_cfg = getattr(self.cfg, "cond", None)
        cond_enabled_cfg = (cond_cfg is not None) and bool(
            getattr(cond_cfg, "enabled", False)
        )

        if not cond_enabled_cfg:
            cond_enabled = False
            prefix_mask = None
            cL_pos = None
        else:
            cL_pos = _sample_cond_len_positions_per_example_continuous(
                self.cfg, B, S, device=self.device
            )
            cond_enabled = bool((cL_pos.max().item() if B > 0 else 0) > 0)
            prefix_mask = (
                _make_prefix_mask_from_lengths(cL_pos, S)
                if cond_enabled
                else None
            )

        noise_prefix = True
        suffix_only_loss = False
        p_uncond = 0.0

        if cond_enabled:
            noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False))
            suffix_only_loss = bool(
                getattr(cond_cfg, "loss_on_suffix_only", True)
            )
            p_uncond = float(getattr(cond_cfg, "p_uncond", 0.0))
            p_uncond = max(0.0, min(1.0, p_uncond))

        # ------------------------------------------------------------------
        # Draw sigma
        # ------------------------------------------------------------------
        sigma = self._draw_sigma(B)
        sigma_exp = (
            sigma.view(-1, 1, 1) if is_cont_tokens else sigma.view(-1, 1)
        )

        # ------------------------------------------------------------------
        # Build xt
        # ------------------------------------------------------------------
        drop_mask = None
        prefix_used_full = None

        if cond_enabled and (not noise_prefix):
            drop_mask = torch.rand(B, device=self.device) < p_uncond

            # Keep an editable prefix tensor only when conditional
            # clean-prefix mode is used.
            prefix_used_full = x0_clean.clone()

            null_full = _make_null_prefix_full(x0_clean, prefix_mask, self.cfg)

            if drop_mask.any():
                if is_cont_tokens:
                    replace = drop_mask.view(B, 1, 1) & prefix_mask.unsqueeze(
                        -1
                    )
                    prefix_used_full[replace] = null_full[replace]
                else:
                    replace = drop_mask.view(B, 1) & prefix_mask
                    prefix_used_full[replace] = null_full[replace]

            del null_full

            # Build xt in-place: xt = sigma * N(0, I) + x0_clean
            xt = torch.empty_like(x0_clean)
            xt.normal_()
            xt.mul_(sigma_exp)
            xt.add_(x0_clean)

            # Clamp prefix positions to chosen clean/null prefix.
            if is_cont_tokens:
                pm = prefix_mask.unsqueeze(-1)
                xt[pm] = prefix_used_full[pm]
            else:
                xt[prefix_mask] = prefix_used_full[prefix_mask]

        else:
            # Unconditional or noisy-prefix mode
            xt = torch.empty_like(x0_clean)
            xt.normal_()
            xt.mul_(sigma_exp)
            xt.add_(x0_clean)

        # In token mode, loss target is x0_target, not x0_clean. If we
        # do not need x0_clean anymore, free it early.
        if is_cont_tokens:
            need_x0_clean_later = cond_enabled and (not noise_prefix)
            if not need_x0_clean_later:
                del x0_clean

        # ------------------------------------------------------------------
        # Self-conditioning
        # ------------------------------------------------------------------
        sc_enabled = bool(getattr(self.cfg.model, "self_condition", False))
        p_sc = float(getattr(self.cfg.train, "self_condition_prob", 0.5))

        x0_hat = None

        if sc_enabled and p_sc > 0.0:
            sc_mask = torch.rand(B, device=self.device) < p_sc
            needs_sc_injection = cond_enabled and (not noise_prefix)

            if sc_mask.any() or needs_sc_injection:
                x0_hat = torch.zeros_like(xt)

            if sc_mask.any():
                with autocast(
                    self.device.type, enabled=amp_enabled, dtype=self.amp_dtype
                ):
                    logits_sc = _model_logits_continuous(
                        self.model,
                        self.cfg,
                        xt,
                        sigma,
                        None,
                    )

                if is_cont_tokens:
                    # Keep detached SC state in xt dtype to avoid fp32
                    # bloat.
                    x0_hat_all = (
                        torch.softmax(logits_sc.float(), dim=-1)
                        .detach()
                        .to(dtype=xt.dtype)
                    )

                    # Avoid torch.where over full tensor when possible.
                    if sc_mask.all():
                        x0_hat.copy_(x0_hat_all)
                    else:
                        x0_hat[sc_mask] = x0_hat_all[sc_mask]
                else:
                    x0_hat_all = (
                        torch.sigmoid(logits_sc.float())
                        .detach()
                        .to(dtype=xt.dtype)
                    )
                    if sc_mask.all():
                        x0_hat.copy_(x0_hat_all)
                    else:
                        x0_hat[sc_mask] = x0_hat_all[sc_mask]

                del logits_sc, x0_hat_all

            if needs_sc_injection:
                if is_cont_tokens:
                    pm = prefix_mask.unsqueeze(-1)
                    x0_hat[pm] = prefix_used_full[pm]
                else:
                    x0_hat[prefix_mask] = prefix_used_full[prefix_mask]

        # prefix_used_full no longer needed after SC injection.
        if prefix_used_full is not None:
            del prefix_used_full

        # ------------------------------------------------------------------
        # Main forward + loss
        # ------------------------------------------------------------------
        with autocast(
            self.device.type, enabled=amp_enabled, dtype=self.amp_dtype
        ):
            logits = _model_logits_continuous(
                self.model,
                self.cfg,
                xt,
                sigma,
                x0_hat,
            )

            loss_mask = None
            if suffix_only_loss and cond_enabled and (not noise_prefix):
                loss_mask = (~prefix_mask).to(dtype=torch.float32)

            loss_target = x0_target if is_cont_tokens else x0_clean

            if self.entropy_compute:
                loss, entropy_metric = self.loss_fn(
                    logits,
                    loss_target,
                    sigma,
                    self.cfg,
                    return_entropy_metric=True,
                    mask=loss_mask,
                )
            else:
                loss = self.loss_fn(
                    logits,
                    loss_target,
                    sigma,
                    self.cfg,
                    return_entropy_metric=False,
                    mask=loss_mask,
                )
                entropy_metric = None

        # Large intermediates no longer needed before backward
        # bookkeeping.
        del logits, xt
        if x0_hat is not None:
            del x0_hat
        if (not is_cont_tokens) and (x0_clean is not None):
            # binary mode still uses x0_clean as loss target until here
            del x0_clean

        # ------------------------------------------------------------------
        # Optim step
        # ------------------------------------------------------------------
        if is_train:
            self.opt.zero_grad(set_to_none=True)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.opt.step()

            self.lr_sched.step()
            self._update_ema()

        # ------------------------------------------------------------------
        # Entropy buffer update
        # ------------------------------------------------------------------
        if self.entropy_compute and (entropy_metric is not None):
            if cond_enabled and (drop_mask is not None):
                keep = ~drop_mask
                if keep.any():
                    self._update_entropy_buffer(
                        sigma[keep], entropy_metric[keep]
                    )
            else:
                self._update_entropy_buffer(sigma, entropy_metric)

        return loss.item()

    def _step_discrete(self, x0, is_train: bool):
        B, S = x0.shape
        x0_flat = x0.to(self.device, non_blocking=True).view(B, -1).long()

        cond_cfg = getattr(self.cfg, "cond", None)
        cond_enabled_cfg = (cond_cfg is not None) and bool(
            getattr(cond_cfg, "enabled", False)
        )

        if cond_enabled_cfg:
            cL_pos = _sample_cond_len_positions_per_example_discrete(
                self.cfg, B, S, self.device
            )
            cond_enabled = bool((cL_pos.max().item() if B > 0 else 0) > 0)
            prefix_mask = (
                _make_prefix_mask_from_lengths(cL_pos, S)
                if cond_enabled
                else None
            )
        else:
            cond_enabled = False
            prefix_mask = None

        noise_prefix = (
            bool(getattr(cond_cfg, "noise_prefix", False))
            if cond_enabled
            else True
        )
        suffix_only_loss = (
            bool(getattr(cond_cfg, "loss_on_suffix_only", True))
            if cond_enabled
            else False
        )
        p_uncond = (
            float(getattr(cond_cfg, "p_uncond", 0.0)) if cond_enabled else 0.0
        )
        p_uncond = max(0.0, min(1.0, p_uncond))

        eps = getattr(self.cfg.diffusion.discrete, "eps", 1e-3)
        t = (1 - eps) * torch.rand(B, device=self.device) + eps
        sigma = self.proc.get_cumulative_noise(t)

        xt = self.proc.sample_xt(x0_flat, t)

        effective_prefix_mask = None
        if cond_enabled and (not noise_prefix):
            drop_mask = torch.rand(B, device=self.device) < p_uncond
            effective_prefix_mask = prefix_mask & (~drop_mask.view(B, 1))
            xt[effective_prefix_mask] = x0_flat[effective_prefix_mask]

        with autocast(
            self.device.type,
            enabled=self.cfg.train.use_fp16,
            dtype=self.amp_dtype,
        ):
            model_scores = self.model(xt, sigma)

            loss_mask = None
            if cond_enabled and (not noise_prefix) and suffix_only_loss:
                loss_mask = (~effective_prefix_mask).to(torch.float32)

            loss = self.loss_fn(
                model_scores,
                x0_flat,
                xt,
                self.proc,
                t,
                self.cfg,
                mask=loss_mask,
            )

        if is_train:
            self.opt.zero_grad(set_to_none=True)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.opt.step()

            self.lr_sched.step()
            self._update_ema()

        return loss.item()

    @torch.compiler.disable
    @torch.no_grad()
    def _validate_epoch(self, step_fn):
        self.model.eval()
        self.ema.apply(self.model)

        # Deterministic validation (opt-in via
        # cfg.train.deterministic_validation): the validation loss is a
        # weighted denoising loss at a *random* sigma and a random noise
        # draw per batch, so with a small validation_max_batches it is a
        # high-variance estimator (it can swing several-fold epoch to
        # epoch). Fixing the sigma / Gaussian-noise stream to a constant
        # seed makes val loss comparable across epochs, so
        # early-stopping and best-checkpoint selection act on real
        # generalization changes rather than sampling noise. The RNG is
        # snapshotted and restored so the training stream is completely
        # unaffected.
        deterministic = bool(
            getattr(self.cfg.train, "deterministic_validation", False)
        )
        rng_snapshot = None
        if deterministic:
            rng_snapshot = self._rng_state()
            vseed = int(getattr(self.cfg.train, "val_seed", 1234))
            random.seed(vseed)
            np.random.seed(vseed)
            torch.manual_seed(vseed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(vseed)

        local_loss = torch.tensor(0.0, device=self.device)
        local_count = torch.tensor(0.0, device=self.device)
        mm_stats = None
        if self.is_multimodal:
            task_weights = getattr(self.cfg.data, "task_weights", {})
            mm_stats = _mm_validation_accumulator(
                self.device, dict(task_weights).keys()
            )

        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            disable=not self.is_master,
        )

        for batch in pbar:
            if (
                0
                < int(getattr(self.cfg.train, "validation_max_batches", 0))
                <= int(local_count.item())
            ):
                break
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            loss = step_fn(x0, is_train=False)
            local_loss += loss
            local_count += 1.0
            if (
                mm_stats is not None
                and self._last_mm_val_diagnostics is not None
            ):
                diagnostics = self._last_mm_val_diagnostics
                _mm_accumulate_validation(
                    mm_stats, diagnostics, diagnostics.get("task_names", [])
                )

        if self.ddp_active:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            if mm_stats is not None:
                for values in mm_stats.values():
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)

        avg_loss = (local_loss / local_count).item()
        val_metrics = (
            _mm_finalize_validation(mm_stats) if mm_stats is not None else {}
        )

        self.ema.restore(self.model)
        if rng_snapshot is not None:
            self._set_rng_state(rng_snapshot)
        return avg_loss, val_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Task-metric selection gate (P0-5)
    # ──────────────────────────────────────────────────────────────────────
    def _maybe_task_score(self) -> dict[str, float]:
        """Score the tasks by SAMPLING, on the frozen subset."""
        if not self.is_multimodal:
            return {}
        node = getattr(self.cfg.train, "task_eval", None)
        every = int(getattr(node, "every_steps", 0) or 0) if node else 0
        if every <= 0:
            return {}
        if (
            self._task_score_last_step is not None
            and (self.global_step - self._task_score_last_step) < every
        ):
            return {}

        from evaluation.proteins.task_score import (
            TaskScoreEvaluator,
        )
        from evaluation.proteins.task_score import (
            finalize as finalize_task_score,
        )

        started = time.time()
        accum: dict[str, float] = {}
        failed = 0.0
        rng_snapshot = self._rng_state()
        # Clear the previous epoch's reason. Without this the field is a
        # one-way latch: one failed gate makes every later epoch look
        # like it failed for that same stale reason, including epochs
        # where the gate simply was not due.
        self._task_score_error = None
        try:
            if self._task_scorer is None:
                dataset = getattr(self.val_loader, "dataset", None)
                if dataset is None:
                    return {}
                self._task_scorer = TaskScoreEvaluator(
                    self.cfg,
                    dataset,
                    self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            # Score the EMA weights, because that is what evaluation and
            # release load. Without the swap the gate would select on a
            # set of weights nothing downstream ever uses.
            #
            # apply()/restore(), not store()/copy_to(): utils.ema.EMA
            # stashes the live weights inside apply() itself. This is
            # the same pair
            # _validate_epoch uses, and getting it wrong cost a
            # 12-minute run -- every rank raised AttributeError, the
            # gate returned nothing, and the
            # checkpointing.metric=task_score branch then killed
            # training.
            self.ema.apply(self.model)
            try:
                accum = self._task_scorer.evaluate(_unwrap_all(self.model))
            finally:
                self.ema.restore(self.model)
        except Exception as exc:
            # Never let the gate kill a run -- but every rank must agree
            # that it failed, or the surviving ranks would hang in the
            # all-reduce below.
            print(
                f"[task_eval] rank {self.rank} skipped: "
                f"{type(exc).__name__}: {exc}"
            )
            # Keep the reason. When checkpointing.metric='task_score'
            # the caller turns an empty result into a fatal error, and
            # that error listing three plausible causes while the real
            # one (an AttributeError from the EMA swap) scrolled past
            # 5,000 steps earlier is how a 12-minute run became a
            # guessing game.
            self._task_score_error = f"{type(exc).__name__}: {exc}"
            failed = 1.0
        finally:
            # The checkpoint-selection evaluator owns a fixed,
            # target-keyed RNG stream and must not advance any training
            # RNG on any rank.
            self._set_rng_state(rng_snapshot)

        keys = list(TaskScoreEvaluator.ACCUMULATOR_KEYS)
        vector = torch.tensor(
            [failed] + [float(accum.get(k, 0.0)) for k in keys],
            dtype=torch.float64,
            device=self.device,
        )
        if self.ddp_active:
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        if float(vector[0]) > 0.0:
            # At least one rank failed. Every rank must now agree on
            # BOTH facts: that the gate failed, and that it failed for a
            # reason (rather than simply not being due).
            # `_task_score_error` is set only on the rank that raised,
            # and the caller uses it to choose between raising and
            # silently skipping -- so without this broadcast the raising
            # rank aborts while the others carry on training, and the
            # survivors then hang in the next collective until the
            # 60-minute PG timeout kills the allocation.
            if self.ddp_active:
                # src=0 is arbitrary but must be fixed; rank 0 may not
                # be the rank that raised, so seed it from any rank that
                # has a reason.
                errs = [None] * self.world_size
                dist.all_gather_object(errs, self._task_score_error)
                first = next((e for e in errs if e), None)
                self._task_score_error = (
                    first or "a rank failed without a reason"
                )
            elif self._task_score_error is None:
                self._task_score_error = "the task evaluator failed"
            return {}  # no partial score

        reduced = {
            k: float(v) for k, v in zip(keys, vector[1:].tolist(), strict=True)
        }
        local_joint_sequences = list(
            getattr(self._task_scorer, "last_joint_sequences", [])
        )
        if self.ddp_active:
            gathered_sequences = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_sequences, local_joint_sequences)
            joint_sequences = [
                sequence
                for rank_sequences in gathered_sequences
                for sequence in (rank_sequences or [])
            ]
        else:
            joint_sequences = local_joint_sequences
        reduced["joint_unique_global"] = float(len(set(joint_sequences)))
        metrics = finalize_task_score(
            reduced, num_steps=self._task_scorer.score_cfg.num_steps
        )
        metrics["task_score_seconds"] = float(time.time() - started)
        self._task_score_last_step = int(self.global_step)
        if self.is_master:
            print(
                f"[task_eval] step {self.global_step}: task_score="
                f"{metrics['task_score']:.4f} (AAR "
                f"{metrics['inverse_folding_aar']:.3f} vs shuffled "
                f"{metrics['inverse_folding_aar_shuffled']:.3f}) in "
                f"{metrics['task_score_seconds']:.0f}s over "
                f"{self.world_size} rank(s)"
            )
        return metrics

    def _log_selection_event(
        self,
        selection_metric: float,
        metric_name: str,
        promoted: bool,
        extra: dict[str, float] | None = None,
    ) -> None:
        """
        Write every selection decision to logs/, not just to the
        checkpoint.
        """
        if not self.is_master:
            return
        try:
            from evaluation.proteins.task_score import append_selection_log

            payload = {
                "global_step": int(self.global_step),
                "metric": metric_name,
                "value": float(selection_metric),
                "best_value": (
                    None
                    if not math.isfinite(self.best_metric)
                    else float(self.best_metric)
                ),
                "mode": self.checkpoint_mode,
                "promoted": bool(promoted),
                "experiment": str(getattr(self.cfg, "experiment", "")),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if extra:
                payload.update({k: float(v) for k, v in extra.items()})
            # Inside the run directory, so the history travels with the
            # run and a deleted run takes its own log with it; and once
            # in the shared repo log, which is the index across runs.
            append_selection_log(self.run_dir / "logs", payload)
            append_selection_log(Path("logs") / "selection", payload)
        except Exception:
            pass  # telemetry must never kill a run

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────
    def _is_better(self, metric: float) -> bool:
        if self.checkpoint_mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def _build_ckpt_state(
        self, epoch: int, rng_states_by_rank=None, entropy_state=None
    ) -> dict:
        raw_model = _unwrap_all(self.model)

        # --- NEW: make EMA checkpoint portable (save shadows on CPU) ---
        ema_sd = self.ema.state_dict()
        ema_sd_cpu = {
            "decay": float(ema_sd["decay"]),
            "shadow": {
                k: v.detach().cpu() for k, v in ema_sd["shadow"].items()
            },
        }
        # If you ever store backups (you currently don't persist them),
        # ignore them: ema_sd_cpu has only decay + shadow, which is all
        # you need.

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": raw_model.state_dict(),
            # --- CHANGED LINE HERE ---
            "ema": ema_sd_cpu,
            "opt": self.opt.state_dict(),
            "lr_sched": self.lr_sched.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_scaler else None,
            "rng_state": self._rng_state(),
            "rng_states_by_rank": rng_states_by_rank,
            "entropy_state": entropy_state,
            "best_metric": self.best_metric,
            "best_ckpts": self.best_ckpts,
            "early_stop_best": self.early_stop_best,
            "early_stop_bad_epochs": self.early_stop_bad_epochs,
        }
        if self._protebit_checkpoint_provenance is not None:
            state["protebit_provenance"] = dict(
                self._protebit_checkpoint_provenance
            )

        # Multimodal resume payload: batch-sampler cursor + collator
        # RNG/counter so a preempted paired run resumes at the exact
        # next batch (gate 5).
        if self.is_multimodal:
            bs = getattr(self.train_loader, "batch_sampler", None)
            coll = getattr(self.train_loader, "collate_fn", None)
            if bs is not None and hasattr(bs, "state_dict"):
                state["mm_sampler"] = bs.state_dict()
            if coll is not None and hasattr(coll, "state_dict"):
                state["mm_collator"] = coll.state_dict()
            state["mm_batch_cursor"] = int(
                getattr(self, "_epoch_batches_done", 0)
            )
            # Carry the realized-mixture counters across chain links.
            # Without this a 94-link chain would reset them every 4
            # hours and the run-end comparison would describe only the
            # final link.
            state["mm_mix_counts"] = dict(self._mm_mix_counts)
            state["mm_mix_total"] = int(self._mm_mix_total)
            state["mm_corpus_microbatches"] = int(self._mm_corpus_microbatches)
            state["mm_replay_microbatches"] = int(self._mm_replay_microbatches)
            state["mm_active_phase"] = int(self._mm_active_phase)
            state["mm_replay_batches_drawn_epoch"] = int(
                getattr(self, "_replay_batches_drawn_epoch", 0)
            )

        return state

    def _save_ckpt(self, epoch: int, val_metric: float, metric_name: str = ""):
        if self.is_multimodal and self._accum_counter != 0:
            raise RuntimeError(
                "refusing to checkpoint inside an accumulation window"
            )
        rng_states = self._collect_rng_states()
        entropy_state = self._entropy_state() if self.is_master else None
        error = None
        if self.is_master:
            try:
                self._save_ckpt_master(
                    epoch, val_metric, metric_name, rng_states, entropy_state
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._raise_distributed_checkpoint_error(error, "selection")

    def _save_ckpt_master(
        self,
        epoch: int,
        val_metric: float,
        metric_name: str,
        rng_states,
        entropy_state,
    ) -> None:
        """Rank-0 filesystem part of _save_ckpt; caller broadcasts failures."""
        new_best = False
        new_best_path = None

        if self.save_top_k > 0 and self._is_better(val_metric):
            new_best = True
            self.best_metric = float(val_metric)

            name = f"epoch={epoch:04d}-val={val_metric:.4f}"
            new_best_path = self._checkpoint_path(name)
            self.best_ckpts.append(
                {
                    "path": new_best_path.name,
                    "metric": float(val_metric),
                    "epoch": int(epoch),
                }
            )

            reverse = self.checkpoint_mode == "max"
            self.best_ckpts.sort(key=lambda d: d["metric"], reverse=reverse)

            while len(self.best_ckpts) > self.save_top_k:
                worst = self.best_ckpts.pop(-1)
                try:
                    os.remove(self.ckpt_dir / worst["path"])
                except FileNotFoundError:
                    pass

            print(
                f"✨ New best model at epoch {epoch} (val_metric="
                f"{val_metric:.4f})"
            )

        state = self._build_ckpt_state(epoch, rng_states, entropy_state)

        if self.save_last:
            tmp_path = self._checkpoint_path("last.tmp")
            final_path = self._checkpoint_path("last")
            torch.save(state, tmp_path)
            os.replace(tmp_path, final_path)

        if self.save_top_k > 0 and new_best and new_best_path is not None:
            _atomic_torch_save(state, new_best_path)
            _atomic_torch_save(state, self._checkpoint_path("best"))
            # A small sidecar next to best.pt so a launcher, an
            # evaluation driver or the next stage's warm start can find
            # out WHAT best means without loading a multi-gigabyte
            # checkpoint.
            try:
                with (self.ckpt_dir / "best.json").open("w") as handle:
                    json.dump(
                        {
                            "path": "best.pt",
                            "epoch": int(epoch),
                            "global_step": int(self.global_step),
                            "metric": metric_name or "unknown",
                            "value": float(val_metric),
                            "mode": self.checkpoint_mode,
                            "experiment": str(
                                getattr(self.cfg, "experiment", "")
                            ),
                            "updated_at_iso": time.strftime(
                                "%Y-%m-%dT%H:%M:%S%z"
                            ),
                        },
                        handle,
                        indent=2,
                    )
            except OSError:
                pass

        self._log_selection_event(
            val_metric, metric_name or "unknown", new_best
        )

    def _write_progress(self, epoch: int, *, strict: bool = False) -> None:
        """Record run progress next to the checkpoints, as small JSON."""
        if not self.is_master:
            return
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(epoch),
            "total_steps": int(getattr(self.cfg.optim, "total_steps", 0) or 0),
            "best_metric": (
                None
                if not math.isfinite(self.best_metric)
                else float(self.best_metric)
            ),
            "updated_at": time.time(),
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "world_size": int(self.world_size),
        }
        path = self.ckpt_dir / "progress.json"
        tmp = self.ckpt_dir / "progress.json.tmp"
        try:
            with tmp.open("w") as f:
                json.dump(payload, f, indent=2)
                if strict:
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            if strict:
                raise
            pass  # never let ordinary telemetry kill a run

    def _write_training_complete(
        self,
        *,
        epoch: int,
        reason: str,
        metric_name: str,
        selection_metric: float,
    ) -> None:
        """
        Atomically mark a deliberately terminal run below its step
        ceiling.
        """
        error = None
        if self.is_master:
            try:
                early = getattr(self.cfg.train, "early_stopping", None)
                if reason != "early_stopping" or early is None:
                    raise ValueError(
                        f"unsupported terminal reason: {reason!r}"
                    )
                expected_rule = (
                    "joint_task_score_no_improvement_6x5k_after_80pct_v1"
                )
                actual_rule = str(getattr(early, "rule_id", ""))
                if actual_rule != expected_rule:
                    raise ValueError(
                        f"unsupported terminal plateau rule: {actual_rule!r}"
                    )
                if (
                    int(getattr(self.cfg.optim, "total_steps", 0) or 0)
                    != 300_000
                    or int(getattr(early, "warmup_steps", 0)) != 240_000
                    or int(getattr(early, "patience_epochs", 0)) != 6
                    or float(getattr(early, "min_delta", -1.0)) != 0.0
                    or int(getattr(self.cfg.train, "steps_per_epoch", 0))
                    != 5_000
                    or metric_name != "task_score"
                    or self.checkpoint_mode != "max"
                ):
                    raise ValueError(
                        "terminal plateau settings differ from the "
                        "preregistered ProteBit-1B Stage-1 rule"
                    )
                # The launcher and evaluation gate require progress.json
                # to agree exactly with the terminal marker.  Selection
                # checkpoints do not otherwise refresh that small
                # sidecar, so make this particular write authoritative
                # and fail closed.
                self._write_progress(epoch, strict=True)
                payload = {
                    "schema_version": 1,
                    "status": "complete",
                    "reason": reason,
                    "global_step": int(self.global_step),
                    "epoch": int(epoch),
                    "maximum_optimizer_steps": int(
                        getattr(self.cfg.optim, "total_steps", 0) or 0
                    ),
                    "selection_metric_name": metric_name,
                    "selection_metric": float(selection_metric),
                    "best_metric": float(self.best_metric),
                    "checkpoint": "best.pt",
                    "plateau_rule": {
                        "rule_id": str(getattr(early, "rule_id", "")),
                        "not_before_optimizer_step": int(
                            getattr(early, "warmup_steps", 0)
                        ),
                        "patience_validations": int(
                            getattr(early, "patience_epochs", 0)
                        ),
                        "minimum_improvement": float(
                            getattr(early, "min_delta", 0.0)
                        ),
                        "validation_interval_optimizer_steps": int(
                            getattr(self.cfg.train, "steps_per_epoch", 0)
                        ),
                    },
                    "experiment": str(getattr(self.cfg, "experiment", "")),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                    "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                path = self.ckpt_dir / "training_complete.json"
                temporary = self.ckpt_dir / "training_complete.json.tmp"
                with temporary.open("w") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._raise_distributed_checkpoint_error(error, "training-complete")

    def _save_resume_ckpt_now(self, epoch: int) -> None:
        """
        Collect rank state and atomically write an exact-boundary
        last.pt.
        """
        if self.is_multimodal and self._accum_counter != 0:
            raise RuntimeError(
                "refusing to checkpoint inside an accumulation window"
            )
        rng_states = self._collect_rng_states()
        entropy_state = self._entropy_state() if self.is_master else None
        error = None
        if self.is_master:
            try:
                state = self._build_ckpt_state(
                    epoch, rng_states, entropy_state
                )
                if self.save_last:
                    tmp_path = self._checkpoint_path("last.tmp")
                    final_path = self._checkpoint_path("last")
                    torch.save(state, tmp_path)
                    os.replace(tmp_path, final_path)
                self._last_resume_ckpt_time = time.time()
                self._write_progress(epoch)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._raise_distributed_checkpoint_error(error, "resume")

    def _maybe_save_resume_ckpt(self, epoch: int) -> None:
        """
        Save the rolling resume checkpoint (last.pt) on a step OR time
        cadence.
        """
        if not self.resume_interval_enabled or not self._did_optim_step:
            return

        due_by_step = (
            self._next_resume_ckpt_step is not None
            and self.resume_interval_every_steps > 0
            and int(self.global_step) >= int(self._next_resume_ckpt_step)
        )
        due_by_time = self.is_master and (
            self.resume_interval_every_seconds > 0
            and (time.time() - self._last_resume_ckpt_time)
            >= self.resume_interval_every_seconds
        )
        due = bool(due_by_step or due_by_time)
        if self.ddp_active:
            flag = torch.tensor(
                int(due), device=self.device, dtype=torch.int32
            )
            dist.broadcast(flag, src=0)
            due = bool(flag.item())
        if not due:
            return

        self._save_resume_ckpt_now(epoch)

        if due_by_step and self._next_resume_ckpt_step is not None:
            while int(self._next_resume_ckpt_step) <= int(self.global_step):
                self._next_resume_ckpt_step += int(
                    self.resume_interval_every_steps
                )

    def _maybe_save_interval_ckpt(self, epoch: int) -> None:
        """
        Save a checkpoint every N steps if enabled.

        Naming: step=000123456.pt Safe: master-only, no collectives.
        Pruning: if keep_last is None => keep ALL interval checkpoints.
        """
        if (not self.ckpt_interval_enabled) or (not self._did_optim_step):
            return
        if self._next_interval_ckpt_step is None:
            return
        if self.ckpt_interval_every_steps <= 0:
            return

        if int(self.global_step) < int(self._next_interval_ckpt_step):
            return

        rng_states = self._collect_rng_states()
        entropy_state = self._entropy_state() if self.is_master else None
        error = None
        if self.is_master:
            try:
                state = self._build_ckpt_state(
                    epoch, rng_states, entropy_state
                )
                name = f"step={int(self.global_step):09d}"
                path = self._checkpoint_path(name)
                _atomic_torch_save(state, path)
                self._interval_ckpt_paths.append(path.name)

                keep_last = self.ckpt_interval_keep_last  # None => keep all
                if keep_last is not None and keep_last > 0:
                    while len(self._interval_ckpt_paths) > keep_last:
                        old = self._interval_ckpt_paths.pop(0)
                        try:
                            os.remove(self.ckpt_dir / old)
                        except FileNotFoundError:
                            pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._raise_distributed_checkpoint_error(error, "interval")

        # Advance on every rank. Rank-0-only advancement makes workers
        # enter the next all-gather while rank 0 skips it, causing a
        # deterministic deadlock.
        while int(self._next_interval_ckpt_step) <= int(self.global_step):
            self._next_interval_ckpt_step += int(
                self.ckpt_interval_every_steps
            )

    # ──────────────────────────────────────────────────────────────────────
    def _update_early_stopping(self, val_metric: float) -> bool:
        cfg = getattr(self.cfg.train, "early_stopping", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return False
        if self.global_step < int(getattr(cfg, "warmup_steps", 0)):
            return False

        min_delta = float(getattr(cfg, "min_delta", 0.0))
        if self.checkpoint_mode == "min":
            improved = val_metric < (self.early_stop_best - min_delta)
        else:
            improved = val_metric > (self.early_stop_best + min_delta)

        if improved:
            self.early_stop_best = float(val_metric)
            self.early_stop_bad_epochs = 0
        else:
            self.early_stop_bad_epochs += 1

        patience = max(1, int(getattr(cfg, "patience_epochs", 1)))
        should_stop = self.early_stop_bad_epochs >= patience
        if self.is_master:
            print(
                f"[early-stop] best={self.early_stop_best:.6f} bad_epochs="
                f"{self.early_stop_bad_epochs}/{patience} min_delta="
                f"{min_delta:g}"
            )
        return should_stop

    # Training Loop
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        if self.is_multimodal:
            step_fn = self._step_multimodal
        elif self.cfg.framework == "continuous_score":
            step_fn = self._step_continuous
        else:
            step_fn = self._step_discrete

        # IMPORTANT: callbacks may include run_on_all_ranks=True (e.g.
        # offline entropy)
        self._run_callbacks("on_train_begin")

        # Lightning-style sanity: run callbacks once before training
        self._maybe_run_sanity()

        target_total_steps = int(getattr(self.cfg.optim, "total_steps", 0))
        stop_training = False
        # True when the walltime deadline or a preemption signal ended
        # this link. The next chain link resumes from the checkpoint
        # just written, so end-of-epoch validation and best.pt selection
        # are skipped: running them would burn the exit margin the
        # deadline reserved.
        preempted = False
        self._run_tp_start_step = int(self.global_step)

        try:
            for epoch in range(self.start_epoch, self.cfg.train.epochs):
                # Critical for DDP: shuffle data differently each epoch
                if self.ddp_active and hasattr(
                    self.train_loader.sampler, "set_epoch"
                ):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                    self.train_loader.batch_sampler.set_epoch(epoch)

                # Multimodal: advance the collator epoch too, and honour
                # a mid-epoch resume cursor for exact next-batch
                # continuation.
                resume_cursor = 0
                if self.is_multimodal:
                    # BEFORE the loader is iterated, so the new task
                    # weights are what the worker processes fork with.
                    # Doing this inside the batch loop would mutate a
                    # collator no worker ever sees.
                    self._mm_refresh_mixture(force=(epoch == self.start_epoch))
                    if getattr(
                        self, "_mm_resume_cursor", 0
                    ) and epoch == getattr(self, "_mm_resume_epoch", -1):
                        resume_cursor = int(self._mm_resume_cursor)
                        self._mm_resume_cursor = 0  # consume once
                    coll = getattr(self.train_loader, "collate_fn", None)
                    if coll is not None and hasattr(coll, "set_epoch"):
                        coll.set_epoch(epoch)
                    # Re-roll the long-chain crop offsets so a chain
                    # longer than the context is seen through a
                    # different window each epoch rather than always
                    # from the same residue.
                    ds = getattr(self.train_loader, "dataset", None)
                    if ds is not None and hasattr(ds, "set_epoch"):
                        ds.set_epoch(epoch)
                    bs = getattr(self.train_loader, "batch_sampler", None)
                    if resume_cursor:
                        if bs is not None and hasattr(bs, "start_batch"):
                            bs.start_batch = resume_cursor
                        if coll is not None and hasattr(coll, "_counter"):
                            coll._counter = resume_cursor
                    # Advance the replay loader's sampler/collator epoch
                    # too, so replay does not draw the same frozen batch
                    # cycle every epoch.
                    if self.replay_loader is not None:
                        rds = getattr(self.replay_loader, "dataset", None)
                        if rds is not None and hasattr(rds, "set_epoch"):
                            rds.set_epoch(epoch)
                        rbs = getattr(
                            self.replay_loader, "batch_sampler", None
                        )
                        if rbs is not None and hasattr(rbs, "set_epoch"):
                            rbs.set_epoch(epoch)
                        rcoll = getattr(self.replay_loader, "collate_fn", None)
                        if rcoll is not None and hasattr(rcoll, "set_epoch"):
                            rcoll.set_epoch(epoch)
                        if resume_cursor and self._mm_resume_replay_cursor:
                            if rbs is None or not hasattr(rbs, "start_batch"):
                                raise RuntimeError(
                                    "replay sampler is not seekable; "
                                    "exact resume is impossible"
                                )
                            rbs.start_batch = int(
                                self._mm_resume_replay_cursor
                            )
                            self._replay_batches_drawn_epoch = int(
                                self._mm_resume_replay_cursor
                            )
                            self._mm_resume_replay_cursor = 0
                        else:
                            self._replay_batches_drawn_epoch = 0
                        self._replay_iter = (
                            None  # rebuild iterator for the new epoch
                        )
                self._epoch_batches_done = resume_cursor
                # Optimizer steps completed in this epoch. On a
                # mid-epoch resume the sampler cursor is in
                # micro-batches, so divide by the accumulation width to
                # recover the optimizer-step count.
                _accum = max(
                    1, int(getattr(self.cfg.train, "grad_accum_steps", 1))
                )
                self._epoch_optim_steps_done = resume_cursor // _accum

                self.model.train()
                train_loss = 0.0
                num_train_batches = 0

                # Start the throughput clock fresh each epoch. Without
                # this the gap between epochs -- validation, the 72 s
                # task_score gate and a multi-GB checkpoint write -- is
                # charged to the first batch of the next epoch as
                # dataloader wait, which is exactly the number this
                # metric exists to make trustworthy.
                self._tp_prev_end = None

                # Disable progress bar on workers
                pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}",
                    leave=True,
                    disable=not self.is_master,
                )

                for batch in pbar:
                    # Time spent blocked on the loader: everything
                    # between the end of the previous iteration and the
                    # arrival of this batch. Measured before the guard
                    # below so an epoch that exits immediately does not
                    # record a phantom wait.
                    _iter_t0 = time.perf_counter()
                    if self._tp_prev_end is not None:
                        _wait = _iter_t0 - self._tp_prev_end
                        self._tp_wait += _wait
                        self._run_tp_wait += _wait

                    # Safety guard in case we resumed exactly at
                    # target_total_steps
                    if (
                        target_total_steps > 0
                        and self.global_step >= target_total_steps
                    ):
                        stop_training = True
                        break

                    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()

                    # Token-budgeted sequence-only replay: with
                    # probability replay_fraction pull a sequence-only
                    # batch (same dict format) so paired training does
                    # not forget the warm-started sequence model (plan
                    # 6.6 step 3). Decision is deterministic per step
                    # for exact resume.
                    did_replay = False
                    if self._mm_should_replay():
                        # Known cost, left in place deliberately: the
                        # paired batch for this micro-step has already
                        # been collated by a worker and is discarded
                        # here, so at a 15-20% replay share that
                        # fraction of collation CPU is wasted. Fixing it
                        # means mixing replay rows in at the collator
                        # level, which changes what
                        # `_epoch_batches_done` -- the sampler resume
                        # cursor -- counts. Not worth coupling to the P0
                        # release; revisit with the shard-format change.
                        batch = self._mm_next_replay_batch()
                        did_replay = True

                    self._mm_record_mixture(batch, did_replay)

                    # Residue-tokens in this micro-batch. The trunk
                    # attends over residues, not bits, so this is the
                    # quantity that makes a tokens/s figure comparable
                    # across chain links and across model sizes. Counted
                    # from residue_mask rather than B*L so a cropped or
                    # padded batch is not over-counted -- and counted
                    # HERE, before step_fn, because the mask is still
                    # the loader's CPU tensor at this point. After the
                    # step it may live on the GPU, where .item() would
                    # insert a device synchronisation on the hot path
                    # once per micro-batch.
                    if isinstance(batch, dict):
                        _rm = batch.get("residue_mask")
                        if _rm is not None and _rm.device.type == "cpu":
                            _tokens = float(_rm.sum().item())
                            self._tp_tokens += _tokens
                            self._run_tp_tokens += _tokens

                    x0 = (
                        batch[0] if isinstance(batch, (list, tuple)) else batch
                    )
                    loss = step_fn(x0, is_train=True)

                    # global_step counts OPTIMIZER steps (effective
                    # batches), so it only advances on an accumulation
                    # boundary. Non-multimodal step fns leave
                    # _did_optim_step True, so they advance every batch
                    # as before. The sampler cursor
                    # (_epoch_batches_done) advances every micro-batch
                    # for exact-next-batch resume.
                    _busy = time.perf_counter() - _iter_t0
                    self._tp_busy += _busy
                    self._run_tp_busy += _busy
                    self._tp_prev_end = time.perf_counter()

                    if self._did_optim_step:
                        self.global_step += 1
                        self._epoch_optim_steps_done += 1
                        self._tp_optim_steps += 1
                        self._run_tp_optim_steps += 1
                    train_loss += loss
                    self._run_last_train_loss = float(loss)
                    if self._run_first_train_loss is None:
                        self._run_first_train_loss = float(loss)
                    num_train_batches += 1
                    # Per-epoch batch cursor (resume-aware; may start >
                    # 0).
                    self._epoch_batches_done += 1

                    # --------------------------------------------------
                    # PATCH: refresh online entropy schedule during
                    # training
                    # --------------------------------------------------
                    if (
                        self.cfg.framework == "continuous_score"
                        and self.entropy_compute
                        and (not self.entropy_offline_enabled)
                        # The multimodal step now fills the entropy
                        # buffer too, so both paths refresh the
                        # schedule.
                        # _did_optim_step keeps this
                        # to one recompute per optimizer step under grad
                        # accumulation (and stays True on the continuous
                        # path, unchanged there).
                        and self._did_optim_step
                    ):
                        update_every = int(
                            getattr(
                                self.cfg.train,
                                "entropy_update_every_steps",
                                2000,
                            )
                        )
                        if update_every > 0 and (
                            self.global_step % update_every == 0
                        ):
                            self._recompute_entropy_from_buffer()

                    # Interval/Resume checkpointing
                    self._maybe_save_resume_ckpt(epoch)
                    self._maybe_save_interval_ckpt(epoch)
                    # Realized (corpus, task) proportions. This is the
                    # only signal that would reveal a mixture that never
                    # reached the forked collator, so it is emitted on
                    # the hot path rather than only at run end.
                    if self._did_optim_step:
                        self._mm_log_mixture()

                    # --------------------------------------------------
                    # Walltime deadline / preemption.
                    #
                    # Checked only at an optimizer-step boundary,
                    # because the check is a collective and every rank
                    # must reach it at the same step. The guard itself
                    # throttles to one check every
                    # preempt_check_every_steps to keep the extra
                    # all-reduce off the hot path.
                    # --------------------------------------------------
                    if self._did_optim_step and self._deadline.should_stop(
                        self.global_step
                    ):
                        # Barrier first so no rank is still inside a
                        # collective when the master starts a
                        # multi-second save.
                        if self.ddp_active:
                            dist.barrier()
                        self._save_resume_ckpt_now(epoch)
                        if self.ddp_active:
                            dist.barrier()
                        if self.is_master:
                            print(
                                f"[deadline] saved last.pt at step "
                                f"{self.global_step}. Exiting for the next"
                                f" chain link."
                            )
                        stop_training = True
                        preempted = True
                        break

                    if self.is_master:
                        pbar.set_postfix(loss=f"{loss:.4f}")

                        # TB logging throttled (HPC-friendly) Running
                        # (EMA) train loss. The per-step diffusion loss
                        # is very noisy (a fresh random sigma each step
                        # swings the EDM-weighted value by an order of
                        # magnitude), so this smoothed curve is what to
                        # read for the training trend; loss/iter_train
                        # keeps the raw per-step value.
                        self._loss_ema = (
                            loss
                            if self._loss_ema is None
                            else 0.98 * self._loss_ema + 0.02 * loss
                        )

                        if self.tb_scalar_every_steps > 0 and (
                            self.global_step % self.tb_scalar_every_steps == 0
                        ):
                            self.writer.add_scalar(
                                "loss/iter_train", loss, self.global_step
                            )
                            self.writer.add_scalar(
                                "loss/iter_train_smooth",
                                self._loss_ema,
                                self.global_step,
                            )
                            lr = self.opt.param_groups[0]["lr"]
                            self.writer.add_scalar(
                                "learning_rate", lr, self.global_step
                            )
                            self.writer.add_scalar(
                                "ema/decay", self.ema.decay, self.global_step
                            )

                            self._log_wandb(
                                {
                                    "loss/iter_train": loss,
                                    "learning_rate": lr,
                                }
                            )

                            # Throughput, reset at every emission so
                            # each point is a rate over the interval
                            # rather than a run-average that flattens as
                            # the run goes on. dataloader_wait_frac is
                            # the actionable one: if it is near zero,
                            # more workers buy nothing and the model is
                            # compute-bound.
                            _span = self._tp_busy + self._tp_wait
                            if _span > 0.0 and self._tp_optim_steps > 0:
                                perf = {
                                    "perf/ms_per_optim_step": 1000.0
                                    * _span
                                    / self._tp_optim_steps,
                                    (
                                        "perf/optim_steps_per_s"
                                    ): self._tp_optim_steps / _span,
                                    "perf/dataloader_wait_frac": self._tp_wait
                                    / _span,
                                    # Per GPU, so the number is
                                    # comparable between a 1-node and a
                                    # 2-node run.
                                    (
                                        "perf/residue_tokens_per_s_per_gpu"
                                    ): self._tp_tokens / _span,
                                }
                                if self.device.type == "cuda":
                                    perf["perf/peak_mem_gib"] = (
                                        torch.cuda.max_memory_allocated(
                                            self.device
                                        )
                                        / (1024**3)
                                    )
                                for k, v in perf.items():
                                    self.writer.add_scalar(
                                        k, v, self.global_step
                                    )
                                self._log_wandb(perf)
                            self._tp_wait = 0.0
                            self._tp_busy = 0.0
                            self._tp_tokens = 0.0
                            self._tp_optim_steps = 0

                            # Per-modality diagnostics for the paired
                            # path.
                            if self.is_multimodal and self._last_mm_components:
                                components = self._last_mm_components
                                mm = {
                                    f"multimodal/{k}": v
                                    for k, v in components.items()
                                }
                                mm["multimodal/replay_step"] = float(
                                    did_replay
                                )
                                if self._trunk_group_active:
                                    mm["multimodal/trunk_lr"] = float(
                                        self.opt.param_groups[1]["lr"]
                                    )
                                for k, v in mm.items():
                                    self.writer.add_scalar(
                                        k, v, self.global_step
                                    )
                                self._log_wandb(mm)

                        # Optional step-based sync (usually keep 0 on
                        # HPC)
                        if (
                            self.tb is not None
                            and self.tb_sync_every_steps > 0
                            and (
                                self.global_step % self.tb_sync_every_steps
                                == 0
                            )
                        ):
                            self.tb.maybe_sync(
                                step=self.global_step, epoch=epoch
                            )

                    if (
                        target_total_steps > 0
                        and self.global_step >= target_total_steps
                    ):
                        stop_training = True
                        break
                    steps_per_epoch = int(
                        getattr(self.cfg.train, "steps_per_epoch", 0)
                    )
                    # P0-7. The epoch budget must be counted in
                    # OPTIMIZER steps, because that is the unit the LR
                    # schedule advances in.
                    #
                    # This used to compare against _epoch_batches_done,
                    # which counts micro-batches. With
                    # grad_accum_steps=k an epoch therefore ended after
                    # steps_per_epoch/k optimizer steps, so a run
                    # reached only (epochs * steps_per_epoch / k) of its
                    # configured total_steps. The 400M config reached
                    # 200,000 of 300,000: the cosine stopped at progress
                    # 0.667 with no anneal at all, and the run exited
                    # normally so nothing flagged it.
                    #
                    # _epoch_batches_done keeps its micro-batch meaning; it is
                    # the sampler cursor that makes an exact mid-epoch
                    # resume possible and must not change units.
                    if (
                        steps_per_epoch > 0
                        and self._epoch_optim_steps_done >= steps_per_epoch
                    ):
                        break

                # Preempted or out of walltime: last.pt is already
                # written and the next link resumes from it. Leave now
                # rather than starting a validation pass we cannot
                # finish.
                if preempted:
                    break

                # If we did not process any batch in this epoch, stop
                # cleanly
                if num_train_batches == 0:
                    if self.is_master:
                        print(
                            f"Reached target total_steps="
                            f"{target_total_steps}. Stopping training."
                        )
                    break

                # Average losses using the actual number of processed
                # batches
                avg_train_loss = train_loss / max(1, num_train_batches)

                # Validation returns the legacy EDM loss plus optional
                # multimodal diagnostics. Configured selection metrics
                # drive best.pt without changing the historical
                # loss/epoch_val series.
                avg_val_loss, val_metrics = self._validate_epoch(step_fn)
                self._run_last_val_loss = float(avg_val_loss)
                checkpoint_cfg = getattr(self.cfg.train, "checkpointing", None)
                checkpoint_metric = str(
                    getattr(checkpoint_cfg, "metric", "edm_loss")
                    if checkpoint_cfg is not None
                    else "edm_loss"
                ).lower()

                # P0-5. Selection on a denoising metric is measurably
                # anti-correlated with task quality here: on an
                # identical protocol `last.pt` scored 37.4%
                # inverse-folding AAR against `best.pt` at 19.2%.
                # `task_score` instead SAMPLES with the bitstream
                # diffusion sampler and scores the tasks. Both denoising
                # metrics stay logged so the three can be compared until
                # they are shown to agree.
                task_metrics = self._maybe_task_score()
                if task_metrics:
                    val_metrics.update(
                        {f"task_eval/{k}": v for k, v in task_metrics.items()}
                    )

                # `None` means "make no selection decision this epoch".
                # The gate is expensive (measured: 72 s across 8 ranks)
                # so it is allowed to run on a slower cadence than the
                # epoch loop -- and when it has not run, the correct
                # behaviour is to leave best.pt alone, not to crash and
                # not to re-promote on a stale value.
                #
                # This is exactly what killed run 5790470:
                # task_eval.every_steps was 25,000 against
                # steps_per_epoch of 5,000, so epoch 1 gated fine and
                # epoch 2 raised.
                selection_metric = None
                if checkpoint_metric == "task_score":
                    if task_metrics:
                        selection_metric = task_metrics["task_score"]
                    elif self._task_score_error:
                        # A gate that RAN and FAILED is a real error:
                        # silently skipping selection would hide a
                        # broken evaluator for the rest of the run.
                        raise RuntimeError(
                            "checkpointing.metric=task_score and the task "
                            "evaluator raised: " + self._task_score_error
                        )
                    elif self.is_master:
                        nxt = (
                            int(getattr(node_te, "every_steps", 0) or 0)
                            if (
                                node_te := getattr(
                                    self.cfg.train, "task_eval", None
                                )
                            )
                            else 0
                        )
                        print(
                            f"[task_eval] not due at step "
                            f"{self.global_step} (every {nxt:,}); best.pt "
                            f"unchanged this epoch"
                        )
                elif checkpoint_metric == "balanced_mse":
                    if "validation/balanced_mse" not in val_metrics:
                        raise RuntimeError(
                            "checkpointing.metric=balanced_mse requires "
                            "multimodal validation diagnostics"
                        )
                    selection_metric = val_metrics["validation/balanced_mse"]
                elif checkpoint_metric in {
                    "edm",
                    "edm_loss",
                    "loss/epoch_val",
                }:
                    selection_metric = avg_val_loss
                else:
                    raise ValueError(
                        f"Unknown checkpointing.metric={checkpoint_metric!r}"
                    )
                early_stop_requested = (
                    self._update_early_stopping(selection_metric)
                    if selection_metric is not None
                    else False
                )

                if self.is_master:
                    self.writer.add_scalar(
                        "loss/epoch_train", avg_train_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "loss/epoch_val", avg_val_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "training/epoch_index", epoch, self.global_step
                    )
                    if selection_metric is not None:
                        self.writer.add_scalar(
                            "validation/selection_score",
                            selection_metric,
                            self.global_step,
                        )
                    if self.device.type == "cuda":
                        peak_gib = torch.cuda.max_memory_allocated(
                            self.device
                        ) / (1024**3)
                        self.writer.add_scalar(
                            "system/peak_cuda_memory_gib",
                            peak_gib,
                            self.global_step,
                        )
                        print(f"Peak CUDA memory: {peak_gib:.2f} GiB")
                    for key, value in sorted(val_metrics.items()):
                        self.writer.add_scalar(key, value, self.global_step)

                    print(
                        (
                            f"Epoch {epoch + 1}: Train Loss = "
                            f"{avg_train_loss:.4f}, Val Loss = "
                            f"{avg_val_loss:.4f}, Selection ("
                            f"{checkpoint_metric}) = "
                        )
                        + (
                            "not evaluated"
                            if selection_metric is None
                            else f"{selection_metric:.6f}"
                        )
                    )

                    self._log_wandb(
                        {
                            "loss/epoch_train": avg_train_loss,
                            "loss/epoch_val": avg_val_loss,
                            **(
                                {
                                    (
                                        "validation/selection_score"
                                    ): selection_metric
                                }
                                if selection_metric is not None
                                else {}
                            ),
                            **val_metrics,
                            "epoch": epoch,
                        }
                    )

                # Run callbacks (some may request running on all ranks)
                self._run_callbacks("on_epoch_end", epoch)

                # Every rank participates so the checkpoint contains
                # exact rank-specific RNG streams. Only rank 0 performs
                # file I/O.
                if selection_metric is None:
                    self._save_resume_ckpt_now(epoch)
                else:
                    self._save_ckpt(epoch, selection_metric, checkpoint_metric)
                terminal_early_stop = (
                    early_stop_requested
                    and str(
                        getattr(
                            getattr(self.cfg.train, "early_stopping", None),
                            "rule_id",
                            "",
                        )
                    )
                    == "joint_task_score_no_improvement_6x5k_after_80pct_v1"
                )
                if terminal_early_stop:
                    self._write_training_complete(
                        epoch=epoch,
                        reason="early_stopping",
                        metric_name=checkpoint_metric,
                        selection_metric=float(selection_metric),
                    )

                # Flush TB buffers and sync staged logs -> run_dir
                # (master only)
                if self.is_master and self.tb is not None:
                    self.tb.flush()
                    if self.tb_sync_every_epochs > 0 and (
                        (epoch + 1) % self.tb_sync_every_epochs == 0
                    ):
                        self.tb.maybe_sync(step=self.global_step, epoch=epoch)

                # NOTE: Old epoch-end entropy recomputation removed on
                # purpose. The schedule is now updated online inside the
                # batch loop.

                # Barrier to keep epochs roughly synced (good practice)
                if self.ddp_active:
                    dist.barrier()

                if stop_training or early_stop_requested:
                    if self.is_master:
                        if early_stop_requested:
                            print(
                                "Early stopping: validation did not "
                                "improve beyond min_delta for the "
                                "configured patience."
                            )
                        else:
                            print(
                                f"Reached target total_steps="
                                f"{target_total_steps}. Stopping training."
                            )
                    break

        except KeyboardInterrupt:
            if self.is_master:
                print(
                    "\n⛔ Training interrupted by user "
                    "(KeyboardInterrupt). Attempting clean shutdown..."
                )
            raise

        finally:
            # Realized-vs-configured mixture, reported unconditionally:
            # a run that was preempted is exactly the case where this is
            # most likely to be wrong and least likely to be checked.
            if self.is_multimodal:
                try:
                    self._mm_assert_mixture()
                except Exception:
                    pass

            write_throughput = (
                os.environ.get("PROTEBIT_WRITE_THROUGHPUT_SUMMARY", "0") == "1"
            )
            global_throughput_tokens = int(self._run_tp_tokens)
            local_peak_memory_fraction = 0.0
            if self.device.type == "cuda":
                local_peak = int(torch.cuda.max_memory_allocated(self.device))
                local_total = int(
                    torch.cuda.get_device_properties(self.device).total_memory
                )
                local_peak_memory_fraction = (
                    float(local_peak / local_total) if local_total else 0.0
                )
            if write_throughput and self.ddp_active:
                token_total = torch.tensor(
                    float(global_throughput_tokens),
                    dtype=torch.float64,
                    device=self.device,
                )
                memory_max = torch.tensor(
                    local_peak_memory_fraction,
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.all_reduce(token_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(memory_max, op=dist.ReduceOp.MAX)
                global_throughput_tokens = int(token_total.item())
                peak_memory_fraction_all_ranks = float(memory_max.item())
            else:
                peak_memory_fraction_all_ranks = local_peak_memory_fraction

            if self.is_master and write_throughput:
                try:
                    span = self._run_tp_busy + self._run_tp_wait
                    steps = int(self._run_tp_optim_steps)
                    if span <= 0 or steps <= 0:
                        raise RuntimeError(
                            "no measured optimizer-step interval"
                        )
                    total_memory = 0
                    peak_memory = 0
                    if self.device.type == "cuda":
                        peak_memory = int(
                            torch.cuda.max_memory_allocated(self.device)
                        )
                        total_memory = int(
                            torch.cuda.get_device_properties(
                                self.device
                            ).total_memory
                        )
                    config_path = Path(os.environ.get("PROTEBIT_CONFIG", ""))
                    config_sha256 = ""
                    if config_path.is_file():
                        config_sha256 = hashlib.sha256(
                            config_path.read_bytes()
                        ).hexdigest()
                    campaign_stage = os.environ.get(
                        "PROTEBIT_CAMPAIGN_STAGE", ""
                    )
                    pilot_rate = os.environ.get(
                        "PROTEBIT_PILOT_LR",
                        os.environ.get("PROTEBIT_PILOT_TRUNK_LR", ""),
                    )
                    payload = {
                        "schema_version": 1,
                        "config": str(config_path),
                        "config_sha256": config_sha256,
                        "campaign_stage": campaign_stage,
                        "run_tag": os.environ.get("PROTEBIT_RUN_TAG", ""),
                        "seed": int(getattr(self.cfg.train, "seed", 0)),
                        "dataset_release": os.environ.get(
                            "PROTEBIT_DATA_RELEASE_ID", ""
                        ),
                        "data_manifest_sha256": (
                            os.environ.get(
                                "PROTEBIT_PAIRED_MANIFEST_SHA256", ""
                            )
                            if campaign_stage == "joint"
                            else os.environ.get(
                                "PROTEBIT_UNIREF_MANIFEST_SHA256", ""
                            )
                        ),
                        "replay_data_manifest_sha256": (
                            os.environ.get(
                                "PROTEBIT_UNIREF_MANIFEST_SHA256", ""
                            )
                            if campaign_stage == "joint"
                            else "none"
                        ),
                        "runtime_contract_sha256": os.environ.get(
                            "PROTEBIT_RUNTIME_CONTRACT_SHA256", ""
                        ),
                        "data_integrity_receipt": os.environ.get(
                            "PROTEBIT_DATA_INTEGRITY_RECEIPT", ""
                        ),
                        "data_integrity_receipt_sha256": os.environ.get(
                            "PROTEBIT_DATA_INTEGRITY_RECEIPT_SHA256", ""
                        ),
                        "environment_contract": os.environ.get(
                            "PROTEBIT_ENVIRONMENT_CONTRACT", ""
                        ),
                        "environment_contract_sha256": os.environ.get(
                            "PROTEBIT_ENVIRONMENT_CONTRACT_SHA256", ""
                        ),
                        "s0_gate_sha256": os.environ.get(
                            "PROTEBIT_S0_GATE_SHA256", ""
                        ),
                        "s0_checkpoint_sha256": os.environ.get(
                            "PROTEBIT_S0_CHECKPOINT_SHA256", ""
                        ),
                        "pilot_learning_rate": (
                            float(pilot_rate) if pilot_rate else None
                        ),
                        "optimizer_learning_rate": float(self.cfg.optim.lr),
                        "precision": (
                            "fp16"
                            if bool(getattr(self.cfg.train, "use_fp16", False))
                            else "fp32"
                        ),
                        "world_size": int(self.world_size),
                        "start_step": int(self._run_tp_start_step),
                        "end_step": int(self._run_tp_start_step + steps),
                        "measurement_steps": steps,
                        "elapsed_training_seconds": float(span),
                        "optimizer_steps_per_second": float(steps / span),
                        "local_nonpadding_residues": int(self._run_tp_tokens),
                        "global_nonpadding_residues": global_throughput_tokens,
                        "nonpadding_residues_per_second": float(
                            global_throughput_tokens / span
                        ),
                        "peak_memory_bytes_rank0": peak_memory,
                        "total_memory_bytes_rank0": total_memory,
                        "peak_memory_fraction_rank0": (
                            float(peak_memory / total_memory)
                            if total_memory
                            else None
                        ),
                        "peak_memory_fraction_all_ranks": (
                            peak_memory_fraction_all_ranks
                        ),
                        "first_train_loss": self._run_first_train_loss,
                        "last_train_loss": self._run_last_train_loss,
                        "validation_score": self._run_last_val_loss,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                        "telemetry_dir": os.environ.get("PROTEBIT_LOGDIR", ""),
                        "source_log": str(
                            Path(os.environ.get("PROTEBIT_LOGDIR", ""))
                            / "node-0.out"
                        ),
                    }
                    protocol_dir = self.run_dir / "protocol"
                    protocol_dir.mkdir(parents=True, exist_ok=True)
                    path = protocol_dir / "throughput_observation.json"
                    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                    tmp.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n"
                    )
                    os.replace(tmp, path)
                except Exception as exc:
                    print(
                        f"[throughput-summary] failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            # Always attempt to flush/sync/close TB even on exceptions /
            # preemption
            if self.is_master and self.tb is not None:
                try:
                    # (optional) best-effort final flush/sync before
                    # TBManager.close()
                    self.tb.flush()
                    self.tb.maybe_sync(
                        step=self.global_step, epoch=max(self.start_epoch, 0)
                    )
                except Exception:
                    pass
                try:
                    self.tb.close()
                except Exception:
                    pass

            # Always finish wandb on master if it was enabled
            if self.use_wandb and wandb is not None and self.is_master:
                try:
                    wandb.finish()
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    # Entropy schedule helpers (continuous) — wrappers
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def entropy_fifo_push(
        self, sigmas: torch.Tensor, metric: torch.Tensor
    ) -> None:
        """
        Push (sigma, metric) pairs into the FIFO ring buffer on CPU.
        sigmas: [B] or [B,1] metric: [B] or [B,1]
        """
        s = sigmas.detach().flatten().to(dtype=torch.float32, device="cpu")
        m = metric.detach().flatten().to(dtype=torch.float32, device="cpu")

        n = s.numel()
        if n == 0:
            return

        cap = self.entropy_buffer_size
        ptr = self._entropy_buf_ptr

        if n >= cap:
            # keep only last cap items
            s = s[-cap:]
            m = m[-cap:]
            n = cap

        end = ptr + n
        if end <= cap:
            self._entropy_sig_buf[ptr:end] = s
            self._entropy_metric_buf[ptr:end] = m
        else:
            k = cap - ptr
            self._entropy_sig_buf[ptr:cap] = s[:k]
            self._entropy_metric_buf[ptr:cap] = m[:k]
            r = n - k
            self._entropy_sig_buf[0:r] = s[k:]
            self._entropy_metric_buf[0:r] = m[k:]

        self._entropy_buf_ptr = (ptr + n) % cap
        self._entropy_buf_len = min(cap, self._entropy_buf_len + n)

    def _entropy_metric_from_logits(
        self,
        logits_2d: torch.Tensor,  # [B, S] or [B, S']
        target_2d: torch.Tensor,  # [B, S] or [B, S']
        mask: torch.Tensor | None = None,  # [B, S] float/bool or None
    ) -> torch.Tensor:
        """
        Per-example mean squared error in probability space
        (unweighted), optionally masked (suffix-only etc).

        Returns: [B]
        """
        # logits -> probs
        probs = torch.sigmoid(logits_2d.float())
        sq_err = (probs - target_2d.float()) ** 2  # [B,S]

        if mask is None:
            return sq_err.mean(dim=1)

        # accept bool or float masks
        if mask.dtype == torch.bool:
            w = mask.to(dtype=sq_err.dtype)
        else:
            w = mask.to(dtype=sq_err.dtype)

        # weighted mean per example; avoid div-by-zero if a row is fully
        # masked
        num = (sq_err * w).sum(dim=1)  # [B]
        den = w.sum(dim=1).clamp_min(1.0)  # [B]
        return num / den

    def _draw_sigma(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.draw_sigma(bsz)

    def _entropy_paths(self):
        return self.entropy_ctrl.entropy_paths()

    def _save_entropy_tables(self, pdf, cdf, sigmas, edges=None):
        return self.entropy_ctrl.save_entropy_tables(pdf, cdf, sigmas, edges)

    def _load_entropy_tables_if_any(self):
        return self.entropy_ctrl.load_entropy_tables_if_any()

    def _entropy_gamma(self) -> float:
        return self.entropy_ctrl.entropy_gamma()

    def _update_entropy_buffer(
        self, sigma: torch.Tensor, entropy_metric: torch.Tensor
    ):
        return self.entropy_ctrl.update_entropy_buffer(sigma, entropy_metric)

    def _fit_lognormal_to_entropy_profile(self):
        return self.entropy_ctrl.fit_lognormal_to_entropy_profile()

    def _recompute_entropy_from_buffer(self):
        return self.entropy_ctrl.recompute_entropy_from_buffer()

    def _sample_entropy_sigma(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.sample_entropy_sigma(bsz)

    def _sample_entropy_sigma_lognormal(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.sample_entropy_sigma_lognormal(bsz)
