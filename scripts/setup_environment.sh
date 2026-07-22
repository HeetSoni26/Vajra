#!/usr/bin/env bash
set -euo pipefail
conda env create -f environment.yml
pre-commit install
