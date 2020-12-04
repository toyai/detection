#!/bin/bash

pip install pip pylint flake8 "isort[colors]" black mypy --progress-bar off -U

set -xu

black . --check

flake8 . --config .flake8

isort . --color --check

pylint jiance tests setup.py
