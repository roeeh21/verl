#! /bin/bash -ex
cd `dirname $0`
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi
. .venv/bin/activate
pip install -r requirements-precommit.txt
pre-commit install
