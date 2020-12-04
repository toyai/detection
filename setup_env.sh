mamba env update -f environment.yml
conda activate detection
pip install pre-commit flake8 black pylint isort mypy
pre-commit install
