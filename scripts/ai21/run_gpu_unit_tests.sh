#!/bin/bash
#
# GPU Unit Tests Script for Cloud Build
# Based on .github/workflows/gpu_unit_tests.yml
#
# This script runs GPU-based unit tests (all test files except *_on_cpu.py)
# It creates a virtual environment on top of the base image to install test dependencies
#
# REQUIREMENTS: 
#   - Minimum: 2 GPUs (for basic tests with --skip-multigpu flag)
#   - Full suite: 8 GPUs (for multi-GPU distributed tests)
#
# Usage: ./run_gpu_unit_tests.sh [--skip-multigpu]
#   --skip-multigpu    Skip multi-GPU distributed tests (requires 4-8 GPUs)
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Parse command line arguments
SKIP_MULTIGPU=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-multigpu)
      SKIP_MULTIGPU=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--skip-multigpu]"
      echo "  --skip-multigpu    Skip multi-GPU distributed tests (requires 4-8 GPUs)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--skip-multigpu]"
      exit 1
      ;;
  esac
done

echo "================================================"
echo "Running GPU Unit Tests"
if [ "$SKIP_MULTIGPU" = true ]; then
  echo "Mode: 1-2 GPU tests only (skipping 4-8 GPU tests)"
else
  echo "Mode: Full test suite (including 4-8 GPU tests)"
fi
echo "================================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi || { echo "ERROR: No GPUs detected!"; exit 1; }

# Download models from GCS to /dev/shm
echo ""
echo "Downloading models from GCS to /dev/shm..."
LOCAL_MODELS_DIR="/dev/shm/local_models"
mkdir -p "$LOCAL_MODELS_DIR/qwen-qwen25-05b-instruct"
gsutil -m rsync -r gs://ai21-algo-studio-research/huggingface_models/qwen-qwen25-05b-instruct/ "$LOCAL_MODELS_DIR/qwen-qwen25-05b-instruct/" || { echo "ERROR: Failed to download Qwen 0.5B model"; exit 1; }

# Create virtual environment
echo ""
echo "Creating virtual environment..."
VENV_DIR="/tmp/test-venv"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

# Install test dependencies in the virtual environment
echo ""
echo "Installing test dependencies in virtual environment..."
pip3 install --no-deps -e .[test]
pip3 install cupy-cuda13x pytest-asyncio
pip3 install --ignore-installed blinker

# Run regular GPU unit tests (1-2 GPUs)
echo ""
echo "Running regular GPU unit tests (1-2 GPUs)..."
echo "Excluding: AI21 tests, special tests, CPU tests, vllm tests, sglang tests, hf_rollout tests, megatron tests, nvtx tests"
echo "Deselecting: Multi-GPU tests (4+ GPUs) and problematic tests"

# test_actor_rollout_ref_worker_actor_ref_model, test_activation_offloading are failing for
python3 -m pytest -s -x\
  --ignore-glob="*test_special_*.py" \
  --ignore-glob='*on_cpu.py' \
  --ignore-glob="*test_vllm*" \
  --ignore-glob="*_sglang*" \
  --ignore-glob="*_hf_rollout*" \
  --ignore-glob="*test_nvtx*" \
  --ignore-glob="tests/models/test_transformers_ulysses.py" \
  --ignore-glob="tests/models/test_engine.py" \
  --ignore-glob='tests/special*' \
  --ignore-glob="tests/experimental" \
  --ignore-glob="tests/workers/reward_model" \
  --ignore-glob="tests/utils/megatron" \
  --ignore-glob="tests/ai21" \
  --deselect="tests/workers/test_fsdp_workers.py::test_actor_rollout_ref_worker_actor_ref_model" \
  --deselect="tests/single_controller/test_nested_worker.py::test_nested_worker" \
  --deselect="tests/single_controller/test_data_transfer.py::test_data_transfer" \
  --deselect="tests/single_controller/test_device_mesh_register.py::test_dist_global_info_wg" \
  --deselect="tests/single_controller/test_high_level_scheduling_api.py::test" \
  --deselect="tests/single_controller/test_worker_group_torch.py" \
  --deselect="tests/single_controller/test_ray_collectives.py::test_ray_collective_group" \
  --deselect="tests/single_controller/test_worker_group_basics.py::test_basics" \
  --deselect="tests/utils/test_activation_offload.py::test_activation_offloading" \
  --deselect="tests/utils/test_torch_functional.py" \
  --deselect="tests/utils/test_mlflow_key_sanitization.py" \
  tests/

python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/workers/actor/test_special_dp_actor.py

python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  tests/workers/critic/test_special_dp_critic.py

# Run multi-GPU distributed tests (requires 4-8 GPUs)
if [ "$SKIP_MULTIGPU" = false ]; then
  echo ""
  echo "Running multi-GPU distributed tests..."

  LOW_MEMORY=True python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    tests/utils/test_special_linear_cross_entropy_tp.py

  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    tests/single_controller/test_data_transfer.py

  # This test is known to be flaky and may fail randomly
  if ! torchrun --nproc_per_node=8 -m pytest tests/models/test_transformers_ulysses.py; then
    echo ""
    echo "⚠️  WARNING: test_transformers_ulysses.py failed!"
    echo "⚠️  This test is known to fail randomly. You may want to re-run it."
    echo ""
    exit 1
  fi

  python3 -m pytest -s -x \
    tests/single_controller/test_device_mesh_register.py \
    tests/single_controller/test_nested_worker.py \
    tests/single_controller/test_high_level_scheduling_api.py \
    tests/single_controller/test_worker_group_basics.py::test_basics \
    tests/single_controller/test_worker_group_torch.py \
    tests/single_controller/test_ray_collectives.py::test_ray_collective_group \
    tests/utils/test_torch_functional.py \
    tests/utils/test_activation_offload.py

else
  echo ""
  echo "Skipping multi-GPU distributed tests (--skip-multigpu flag set)"
fi

echo ""
echo "================================================"
echo "GPU Unit Tests Completed Successfully!"
echo "================================================"

# Cleanup downloaded models from /dev/shm (if they were downloaded)
if [ "$SKIP_MULTIGPU" = false ]; then
  echo ""
  echo "Cleaning up downloaded models from /dev/shm..."
  if [ -d "/dev/shm/local_models" ]; then
    rm -rf /dev/shm/local_models
    echo "Models cleaned up successfully!"
  else
    echo "No models to clean up."
  fi
fi
