#!/bin/bash

set -xu

black .

flake8 . --config .flake8

isort . --color

pylint toydet
