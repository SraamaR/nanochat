#!/bin/bash

# all the setup stuff

export OMP_NUM_THREADS=1

# For running on unsupported RDNA2 GPUs like gfx1035
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Workaround for PyTorch download
# export TMPDIR=/home/user/Documents/Projets/nanochat/.tmp

# Intermediate artifacts directory is in ./.cache
export NANOCHAT_BASE_DIR="$(pwd)/.cache"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || python3 -m venv .venv

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# install the repo dependencies
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d6 bash rocm_pretrain.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Each shard is ~250M chars so ~52M tokens
python -m nanochat.dataset -n 16
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=32768
python -m scripts.tok_eval

# Number of processes/GPUs to use
NPROC_PER_NODE=1

python -m scripts.base_train \
    --depth=6 \
    --device_batch_size=1 \
    --total_batch_size=131072 \
    --eval_every=50 \
    --eval_tokens=524288 \
    --core_metric_every=-1 \
    --core_metric_max_per_task=12 \
    --sample_every=200 \
    --num_iterations=2400 \
    --dtype=fp16 \
    --weight_decay=0.71 \
    --run=$WANDB_RUN
python -m scripts.base_loss --device_batch_size=1 --split_tokens=524288
python -m scripts.base_eval --max-per-task=16

# midtraining
#python -m scripts.mid_train \
#    --max_seq_len=1024 \
#    --device_batch_size=1 \
#    --eval_every=50 \
#    --eval_tokens=4096 \
#    --total_batch_size=1024 \
#    --num_iterations=100

# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
#python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
#python -m scripts.chat_sft \
#    --device_batch_size=1 \
#    --target_examples_per_step=4 \
#    --num_iterations=100 \
#    --eval_steps=4 \
#    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
