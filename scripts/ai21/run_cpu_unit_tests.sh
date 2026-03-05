#!/bin/bash
#
# CPU Unit Tests Script for Bitbucket CI
# Based on .github/workflows/cpu_unit_tests.yml
#
# This script runs CPU-based unit tests (test files ending with *_on_cpu.py)
# It creates a virtual environment on top of the base image to install test dependencies
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

echo "================================================"
echo "Running CPU Unit Tests"
echo "================================================"

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
VENV_DIR="/tmp/test-venv"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

# Step 2: Install test dependencies in the virtual environment
echo ""
echo "Step 2: Installing test dependencies in virtual environment..."
pip install -e .[test,geo]

# Step 3: Download datasets
echo ""
echo "Step 3: Downloading required datasets..."
mkdir -p ~/verl-data/gsm8k
gsutil cp "gs://ai21-algo-studio-research/huggingface_datasets/verl-team/gsm8k-v0.4.1/*" ~/verl-data/gsm8k
python3 examples/data_preprocess/geo3k.py

# Step 4: Run CPU unit tests
echo ""
echo "Step 4: Running CPU unit tests..."
echo "Test pattern: tests/**/test_*_on_cpu.py"

# Run pytest with pattern matching for CPU tests
pytest -s -x --asyncio-mode=auto tests/**/*_on_cpu.py

# Cleanup
echo ""
echo "Step 5: Cleaning up virtual environment..."
deactivate
rm -rf "$VENV_DIR"

echo ""
echo "================================================"
echo "CPU Unit Tests Completed Successfully!"
echo "================================================"

