# CoBit for proteins

This fork adds protein sequence and structure modeling to
[CoBit](https://github.com/GBATZOLIS/BitstreamDiffusion).
The original text model code and configs are kept.

The protein model uses 5 sequence bits and 13 structure bits per residue.
`configs/proteins/paper.py` contains the 400M paper training settings and
its entropic sampler settings. The model has 396,226,417 parameters.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
./setup.sh train
```

This creates `.venv` from `uv.lock`, using Python 3.11 and PyTorch 2.7.1
with CUDA 12.6 on Linux. Run all commands from the repository root.

## Data and config

The model config is included in this repository. Download the frozen DPLM
tokenizer config and prepare the training data:

```bash
uv run --no-sync python scripts/proteins/download_config.py
uv run --no-sync python scripts/proteins/download_data.py train
```

This prepares paired PDB and Swiss-Prot examples from the
[DPLM dataset](https://huggingface.co/datasets/airkingbd/pdb_swissprot)
and 400,000 replay sequences from
[EvoDiff UniRef50](https://zenodo.org/records/6564798).
Replay preparation downloads the full UniRef50 archive before selecting
sequences. Keep enough disk space for the archive and extracted data.
Prepared data goes in `datasets/`. Add `--dry-run` to preview downloads.

## Train

The paper run used eight H100 GPUs, FP32, 300,000 optimizer steps, and
an effective batch of 512. Run on a machine with suitable GPUs:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/proteins/paper.py --run-tag paper1
```

Checkpoints and training logs go in
`.trentinium/runs/proteins/paper/paper1/`.
Use the same command with `--resume` to continue an interrupted run.
Use a new run tag to start again.

## Evaluate

Install the evaluation dependencies and download the frozen tokenizer,
ESMFold weights, and ProteinMPNN code and weights:

```bash
./setup.sh eval
uv run --no-sync python scripts/proteins/download_data.py eval
```

This also downloads the pinned OpenFold and DPLM source used by the
structure tokenizer. Model downloads are several GB.

Estimate entropy tables from CATH validation, then evaluate a checkpoint:

```bash
checkpoint=.trentinium/runs/proteins/paper/paper1/checkpoints/best.pt
uv run --no-sync python scripts/proteins/evaluate.py entropy --checkpoint "$checkpoint"
uv run --no-sync python scripts/proteins/evaluate.py inverse --checkpoint "$checkpoint"
uv run --no-sync python scripts/proteins/evaluate.py forward --checkpoint "$checkpoint"
uv run --no-sync python scripts/proteins/evaluate.py cogeneration --checkpoint "$checkpoint"
```

Inverse folding uses CATH test backbones. Forward folding uses the CAMEO
2022 split from DPLM. Co-generation samples lengths 100, 200, 300, 400,
and 500. The backbone check uses one ProteinMPNN redesign per sample.
These commands use the saved paper sampler: 50 grid points, CFG 5 for
conditional tasks, and entropy-based churn. They do not run a sweep.

Reports go in `.trentinium/results/proteins/`. Use `--out PATH` for separate
reports, `--resume` to continue evaluation, or `--dry-run` to print a command.
The entropy tables stay beside the checkpoint run and are checked against
its hash. Reproducing exact paper scores also requires the original weights
and benchmark subsets; retraining does not guarantee identical scores.

## Upload the protein model

After computing the entropy tables, check the upload file list:

```bash
checkpoint=.trentinium/runs/proteins/paper/paper1/checkpoints/best.pt
uv run --no-sync python scripts/upload_to_hf.py \
  --model protein --repo-id your-name/CoBit-Protein \
  --checkpoint "$checkpoint" --dry-run
```

Set `HF_TOKEN` to a write token or use cached Hugging Face credentials,
then remove `--dry-run` to upload. Add `--private` to create a private repo.
The upload includes only the selected checkpoint, `configs/proteins/paper.py`,
and its four entropy tables plus `entropy_profile.json`. The checkpoint is
uploaded unchanged, including any training state it contains. Checkpoint
and entropy table hashes are checked before upload. Datasets, logs, and
other results are excluded.
Use `--entropy-dir PATH` if the entropy files are outside the checkpoint run.
The original text upload remains available with `--model text`.

Download the Hub files into a local run directory. Pass its
`checkpoints/best.pt` as `--checkpoint` and its `configs/proteins/paper.py`
as `--config` to `scripts/proteins/evaluate.py`. Use the data and model
download commands above to install the external evaluation dependencies.

## Tests

```bash
uv sync --locked --extra eval --group dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest
uv run --no-sync python scripts/check_protein_style.py
```

The style check covers the protein additions and shared modules they extend.
It checks Python style, formatting, and a hard limit of 79 columns.

Documentation, old configs, logs, results, and agent files are stored locally
in `.trentinium/`, which is ignored by git. The upstream license is in `LICENSE`.
