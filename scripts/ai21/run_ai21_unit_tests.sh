#!/bin/bash

set -e
set -u
set -o pipefail

echo "================================================"
echo "Running AI21 Unit Tests"
echo "================================================"

python3 -m pytest -s -x tests/ai21/

echo ""
echo "================================================"
echo "AI21 Tests Completed Successfully!"
echo "================================================"

