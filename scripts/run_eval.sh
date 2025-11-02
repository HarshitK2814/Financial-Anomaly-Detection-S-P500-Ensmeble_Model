#!/bin/bash

# Activate the conda environment
source activate your_environment_name

# Run the evaluation script
python src/evaluation/evaluate.py --config src/experiments/configs/base.yaml

# Deactivate the conda environment
conda deactivate