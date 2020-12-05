#!/bin/bash

set -xu

if [[ "${TORCH_UTILS_COLLECT_ENV}" -eq 1 ]]; then
    python -m torch.utils.collect_env
fi

coverage run --source jiance -m unittest discover -v -s ./tests/ -p "test_*.py"

coverage report
