# train.py
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from ml_collections import config_dict

from trainers import Trainer

# Keys whose value must come from the CURRENT environment on every chain
# link, never from the config.json a previous link wrote.
#
# A run under a 4-hour walltime is 10-90 chained jobs. train.py merges
# the saved config over the freshly imported one whenever last.pt
# exists, which is right for model shape (link 12 must not silently
# change the architecture) and wrong for anything tied to the machine or
# the schedule. If link 1 staged data to a per-job path, link 2 restores
# that dead path and every rank dies at once with FileNotFoundError. The
# same merge also means extending total_steps or fixing a checkpoint
# metric requires hand-editing JSON inside the run directory, with only
# "Resuming experiment using saved config" printed to say so.
RESUME_MUTABLE_PREFIXES = (
    "data.root",
    "data.shard_dir",
    "data.sequence_replay_root",
    "data.num_workers",
    "data.pin_memory",
    "data.persistent_workers",
    "data.prefetch_factor",
    "train.grad_accum_steps",
    "train.num_workers",
    "train.max_wall_seconds",
    "train.exit_margin_seconds",
    "train.checkpoint_every_seconds",
    "train.preempt_check_every_steps",
    "train.epochs",
    "train.steps_per_epoch",
    "train.use_compile",
    "train.compile_mode",
    "optim.total_steps",
    # FULL dotted paths. These were previously the bare segment names
    # "checkpointing" / "task_eval", which never matched: the flattened
    # key is `train.checkpointing.metric`, and
    # "train.checkpointing.metric" does not start with "checkpointing".
    # So the entire checkpointing and task-eval blocks were silently
    # FROZEN across every resume -- a fix to task_eval.every_steps or to
    # checkpointing.metric would apply on link 1 and be reverted from
    # config.json on links 2..94, which is exactly backwards for the two
    # knobs most likely to need correcting mid-run.
    "train.checkpointing",
    "train.task_eval",
    "train.global_loss_normalizer",
    "train.validation_self_condition",
    "train.allow_partial_load",
    "logging",
    "system",
    "evaluation",
)

# Keys that must NOT change across a resume. A mismatch means the
# checkpoint and the config describe different models, and loading would
# either fail loudly or, worse, succeed and quietly invalidate every
# capacity claim about the run.
RESUME_FROZEN_PREFIXES = (
    "model.",
    "train.batch_size",
    "train.budget_schedule_sha256",
    "diffusion.",
)


def _is_distributed():
    """Check if we are running under torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _setup_ddp():
    """Initialize the DDP process group if distributed."""
    if not _is_distributed():
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Bind the device BEFORE init_process_group and pass device_id.
    # Doing it after makes NCCL guess the mapping from the global rank,
    # which it warns about ("Guessing device ID based on global rank.
    # This can cause a hang if rank to GPU mapping is heterogeneous")
    # and which does hang on a node where the mapping is not the
    # identity.
    torch.cuda.set_device(local_rank)

    # The default 20-minute timeout is shorter than a worst-case
    # checkpoint save on a contended shared filesystem with every rank
    # also reading, which makes the whole allocation die with an opaque
    # NCCL error at a predictable interval. 60 minutes costs nothing
    # when things are healthy.
    timeout_min = int(os.environ.get("PROTEBIT_DIST_TIMEOUT_MINUTES", "60"))
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=timeout_min),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    if rank == 0:
        print(
            f"DDP initialized: world_size={world_size}, timeout="
            f"{timeout_min}min"
        )
    return rank, local_rank, world_size


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _flatten(d, prefix=""):
    """Flatten a nested dict into {'a.b.c': value}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _get_dotted(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not hasattr(node, part):
            return None, False
        node = getattr(node, part)
    return node, True


def _update_cfg_from_dict(cfg, d, skip_prefixes=(), prefix=""):
    """Merge saved values into cfg, leaving resume-mutable keys alone."""
    for k, v in d.items():
        dotted = f"{prefix}{k}"
        # Segment-boundary matching only. The old third clause,
        # `dotted.startswith(p)`, matched any key that merely began with
        # the prefix STRING -- so "train.epochs" also captured a
        # hypothetical "train.epochs_extra", and bare "system" captured
        # "system_foo". A prefix list that silently over-matches is how
        # a key becomes mutable by accident and stops being restored on
        # resume.
        if any(
            dotted == p or dotted.startswith(p + ".") for p in skip_prefixes
        ):
            continue
        if isinstance(v, dict):
            if not hasattr(cfg, k) or getattr(cfg, k) is None:
                setattr(cfg, k, config_dict.ConfigDict())
            _update_cfg_from_dict(
                getattr(cfg, k),
                v,
                skip_prefixes=skip_prefixes,
                prefix=f"{dotted}.",
            )
        else:
            setattr(cfg, k, v)


def _assert_frozen_keys_match(cfg, saved_flat, rank):
    """Refuse to resume when a model-shape key changed since the last link."""
    problems = []
    for dotted, saved_value in saved_flat.items():
        if not dotted.startswith(RESUME_FROZEN_PREFIXES):
            continue
        current, found = _get_dotted(cfg, dotted)
        if not found:
            continue
        if (
            isinstance(saved_value, (int, float, str, bool))
            and current != saved_value
        ):
            problems.append(
                f"  {dotted}: checkpoint={saved_value!r} config={current!r}"
            )
    if problems:
        msg = (
            "Refusing to resume: the config changed a key that defines the "
            "model or the optimized objective.\n"
            + "\n".join(problems)
            + "\n\nStart a new experiment name, or revert the change."
        )
        if rank == 0:
            print(msg)
        raise SystemExit(2)


def _apply_run_tag(cfg, tag: str, rank: int) -> None:
    """Give this run its own directory under the config's experiment name."""
    tag = str(tag).strip().strip("/")
    if not tag:
        return
    cfg.experiment = f"{cfg.experiment}/{tag}"
    ev = getattr(cfg, "evaluation", None)
    if ev is None:
        return
    base = str(Path(getattr(cfg, "output_root", "runs")) / cfg.experiment)
    for key, value in (
        ("checkpoint_path", f"{base}/checkpoints/last.pt"),
        ("best_checkpoint_path", f"{base}/checkpoints/best.pt"),
        ("out_dir", f"{base}/protein_eval"),
        ("samples_dir", f"{base}/protein_eval/samples"),
    ):
        if hasattr(ev, key):
            setattr(ev, key, value)
    if rank == 0:
        print(f"[run-tag] experiment -> {cfg.experiment}")


def _guard_accidental_resume(
    run_dir: Path, allow_resume: bool, rank: int
) -> None:
    """Refuse to silently continue an existing run."""
    last = run_dir / "checkpoints" / "last.pt"
    if not last.exists() or allow_resume:
        return
    if rank == 0:
        print(
            f"\nRefusing to start: {last} already exists.\n\nContinuing it"
            f" would append this run to the previous one's step\ncount, LR"
            f" schedule and W&B curve, which is right for a chain link\n"
            f"and wrong for a new run. Pick one:\n\n  continue that run   "
            f"--resume            (submit.sh)\n                      "
            f"PROTEBIT_ALLOW_RESUME=1 (direct)\n  start a new one     "
            f"--run-tag <name>    (submit.sh)\n                      "
            f"PROTEBIT_RUN_TAG=<name> (direct)\n  discard the old one rm "
            f"-rf {run_dir}\n"
        )
    raise SystemExit(3)


def main():
    rank, local_rank, world_size = _setup_ddp()

    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config."
    )
    parser.add_argument(
        "--run-tag",
        default=os.environ.get("PROTEBIT_RUN_TAG", ""),
        help=(
            "Unique sub-directory under runs/<experiment>, so a new run "
            "cannot land on an existing one's checkpoints."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("PROTEBIT_ALLOW_RESUME", "0").lower()
        in ("1", "true", "yes"),
        help="Continue an existing run directory. Without it, a run directory "
        "that already holds last.pt is a hard error.",
    )
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config()
    base_experiment = str(cfg.experiment)

    # Seed override. Decision 17.4 calls for three seeds at 130M, and
    # without this that means three near-duplicate config files whose
    # only difference is one integer -- which then drift. The seed is a
    # frozen resume key, so a chain link picks it up from the saved
    # config and this only ever applies to a run's first link.
    seed_override = os.environ.get("PROTEBIT_SEED", "")
    if seed_override:
        cfg.train.seed = int(seed_override)
        if rank == 0:
            print(f"[seed] cfg.train.seed -> {cfg.train.seed}")

    # The run tag is applied BEFORE anything reads cfg.experiment, so
    # the run directory, the evaluation paths, the checkpoint directory
    # and the W&B run name all agree on which run this is.
    _apply_run_tag(cfg, args.run_tag, rank)

    # Stage-to-stage warm start. The source run directory now carries a
    # tag, so a config cannot name it literally without hardcoding
    # whatever tag the launcher happened to use. The orchestrator passes
    # it instead.
    init_run = os.environ.get("PROTEBIT_INIT_FROM_RUN", "")
    init_mode = os.environ.get("PROTEBIT_INIT_FROM", "")
    is_protebit_s0 = (
        base_experiment == "proteins/protebit_flagship/s0"
        or base_experiment == "proteins/protebit_flagship/pilots/s0"
        or base_experiment.startswith("proteins/protebit_flagship/probes/s0_")
    )
    is_protebit_joint = (
        base_experiment == "proteins/protebit_flagship/joint"
        or base_experiment == "proteins/protebit_flagship/pilots/joint"
        or base_experiment.startswith(
            "proteins/protebit_flagship/probes/joint_"
        )
    )
    is_protebit_production = base_experiment in {
        "proteins/protebit_flagship/s0",
        "proteins/protebit_flagship/joint",
    }
    if is_protebit_production and not os.environ.get(
        "PROTEBIT_1B_GATE_ATTESTATION", ""
    ):
        raise RuntimeError(
            "ProteBit-1B production training requires the campaign launch "
            "attestation; use the one-command orchestrator"
        )
    if is_protebit_s0 and (init_run or init_mode):
        raise RuntimeError(
            "ProteBit-1B Stage-0/probes must start from a fresh exact-1B"
            " initialization; unset PROTEBIT_INIT_FROM_RUN and "
            "PROTEBIT_INIT_FROM"
        )
    if is_protebit_joint and (not init_run or init_mode != "best"):
        raise RuntimeError(
            "ProteBit-1B joint work requires PROTEBIT_INIT_FROM_RUN bound to "
            "this campaign's passed Stage-0 run and PROTEBIT_INIT_FROM=best"
        )
    if init_run:
        cfg.train.init_from_run = init_run
        cfg.train.init_from = init_mode or "best"
        if rank == 0:
            print(
                f"[init_from] {cfg.train.init_from!r} from {init_run} "
                f"(weights only; step count and LR schedule start fresh)"
            )

    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = world_size > 1
    cfg.system.global_rank = rank
    cfg.system.local_rank = local_rank
    cfg.system.world_size = world_size

    run_dir = Path(getattr(cfg, "output_root", "runs")) / cfg.experiment
    saved_cfg_path = run_dir / "config.json"
    last_ckpt = run_dir / "checkpoints" / "last.pt"

    _guard_accidental_resume(run_dir, args.resume, rank)

    if last_ckpt.exists() and saved_cfg_path.exists():
        with open(saved_cfg_path) as f:
            saved_cfg_dict = json.load(f)
        saved_flat = _flatten(saved_cfg_dict)
        _assert_frozen_keys_match(cfg, saved_flat, rank)
        _update_cfg_from_dict(
            cfg, saved_cfg_dict, skip_prefixes=RESUME_MUTABLE_PREFIXES
        )
        if rank == 0:
            # Report what the merge actually CHANGED, not what it might
            # have.
            #
            # On a 90-link chain the merge runs 89 times and each one
            # silently reverts anything not on the mutable list.
            # "Resuming from the saved config" told you it happened; it
            # never told you WHICH of your edits were thrown away, so a
            # config fix that quietly stopped applying after link 1
            # looked exactly like a fix that worked.
            reverted = []
            for dotted, saved_value in saved_flat.items():
                if any(
                    dotted == m or dotted.startswith(m + ".")
                    for m in RESUME_MUTABLE_PREFIXES
                ):
                    continue
                current, found = _get_dotted(cfg, dotted)
                if (
                    found
                    and isinstance(saved_value, (int, float, str, bool))
                    and current != saved_value
                ):
                    reverted.append((dotted, current, saved_value))
            print(f"Resuming '{cfg.experiment}' from {saved_cfg_path}.")
            if reverted:
                print(
                    f"  {len(reverted)} key(s) in your config were "
                    f"OVERRIDDEN by the saved one (they are not "
                    f"resume-mutable):"
                )
                for dotted, cur, saved in reverted[:12]:
                    print(f"    {dotted}: yours={cur!r} -> using={saved!r}")
                if len(reverted) > 12:
                    print(f"    ... and {len(reverted) - 12} more")
                print(
                    "  To make one of these take effect, add it to "
                    "RESUME_MUTABLE_PREFIXES, or start a new run with --new."
                )
            else:
                print("  No config key was overridden; your edits all apply.")
    elif rank == 0:
        print(
            "No previous run with a saved config found; using the Python"
            " config."
        )

    # Data paths follow the environment on every link, because a chained
    # job may be staged to a different node-local directory than the one
    # link 1 used.
    stage_root = os.environ.get("PROTEBIT_DATA_ROOT")
    if stage_root and hasattr(cfg, "data"):
        for key in ("shard_dir", "sequence_replay_root", "root"):
            current = getattr(cfg.data, key, None)
            if not current:
                continue
            name = Path(str(current)).name
            # Try both layouts: the tarball is packed with a leading
            # "datasets/" and extracted with --strip-components=1, but
            # accept the unstripped form too so a hand-made tarball
            # still works.
            for candidate in (
                Path(stage_root) / name,
                Path(stage_root) / "datasets" / name,
            ):
                if candidate.exists():
                    setattr(cfg.data, key, str(candidate))
                    if rank == 0:
                        print(
                            f"  data.{key} -> {candidate}  (node-local stage)"
                        )
                    break
            else:
                # Never fail silently here. A staging miss is invisible
                # at run time -- training simply reads over the shared
                # filesystem and looks fine -- so the only signal is
                # this message.
                if rank == 0:
                    print(
                        f"  WARNING data.{key}: no staged copy of {name!r}"
                        f" under {stage_root}; reading from {current} over"
                        f" the shared filesystem instead. Every rank will "
                        f"read it independently."
                    )

    cfg._config_path = str(Path(args.config).resolve())

    try:
        trainer = Trainer(cfg)
        trainer.train()
    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
