#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=<path>%J.out
#SBATCH --error=<path>-%J.err
#SBATCH --partition=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --exclude=clippy,cyborg,ig-88,megazord
#SBATCH --mem-per-gpu=45G

source ~/.bashrc
set -euo pipefail

REPO_ROOT="/coc/testnvme/xzhang3205/vla-adaptation"
GR00T_ROOT="${REPO_ROOT}/models/gr00t-n1.5"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/slurms"
source "${GR00T_ROOT}/.venv/bin/activate"

export BENCHMARK="${BENCHMARK:-robocasa}"
export TASK_ID="${TASK_ID:-2}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-$((19080 + SLURM_JOB_ID % 1000))}"
export PYTHONPATH="${REPO_ROOT}:${GR00T_ROOT}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/coc/testnvme/xzhang3205/.cache"
export HF_HOME="/coc/testnvme/xzhang3205/huggingface"
export TRANSFORMERS_CACHE="/coc/testnvme/xzhang3205/huggingface"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
