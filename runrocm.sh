#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff

export OMP_NUM_THREADS=1

# For running on unsupported RDNA2 GPUs
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Intermediate artifacts directory is in ./.cache
export NANOCHAT_BASE_DIR="$(pwd)/.cache"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup

# check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it manually."
    exit 1
fi

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || python3 -m venv .venv --system-site-packages
# install the repo dependencies
#uv sync --extra rocm

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

pip install -r requirements.txt

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
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

# check for cargo
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo is not installed. Please install it manually."
    exit 1
fi

# Build the rustbpe Tokenizer
maturin develop --release --manifest-path rustbpe/Cargo.toml

# train tokenizer on ~1B characters
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
#python -m scripts.base_train \
#    --depth=4 \
#    --max_seq_len=1024 \
#    --device_batch_size=1 \
#    --total_batch_size=1024 \
#    --eval_every=1000 \
#    --eval_tokens=4096 \
#    --core_metric_every=1000 \
#    --core_metric_max_per_task=12 \
#    --sample_every=1000 \
#    --num_iterations=1000

# Number of processes/GPUs to use
NPROC_PER_NODE=1

python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=8192 \
    --eval_every=100 \
    --eval_tokens=81920 \
    --core_metric_every=-1 \
    --core_metric_max_per_task=12 \
    --sample_every=500 \
    --num_iterations=2500
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
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
