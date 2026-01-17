#!/bin/bash

# all the setup stuff

export OMP_NUM_THREADS=1

# For running on unsupported RDNA2 GPUs like gfx1035
#export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Workaround for PyTorch download
#mkdir $(pwd)/.tmp
#export TMPDIR="$(pwd)/.tmp"

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
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
python -m scripts.tok_eval

# Number of processes/GPUs to use
NPROC_PER_NODE=1

python -m scripts.base_train \
    --depth=6 \
    --device-batch-size=1 \
    --total-batch-size=131072 \
    --eval-every=50 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --core-metric-max-per-task=12 \
    --sample-every=200 \
    --num-iterations=2400 \
    --window-pattern L \
    --run=$WANDB_RUN
python -m scripts.base_loss --device-batch-size=1 --split-tokens=524288
python -m scripts.base_eval --max-per-task=16

# midtraining
#python -m scripts.mid_train \
#    --max-seq-len=1024 \
#    --device-batch-size=1 \
#    --eval-every=50 \
#    --eval-tokens=4096 \
#    --total-batch-size=1024 \
#    --num-iterations=100

# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
#python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
#python -m scripts.chat_sft \
#    --device-batch-size=1 \
#    --target-examples-per-step=4 \
#    --num-iterations=100 \
#    --eval-steps=4 \
#    --eval-metrics-max-problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
