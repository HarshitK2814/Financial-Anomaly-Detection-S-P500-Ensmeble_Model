#!/bin/bash

# Activate the conda environment
source activate your_environment_name

# Set the paths for data and output directories
DATA_DIR="path/to/your/data"
OUTPUT_DIR="path/to/your/output"

# Run the training script
python src/training/train.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --config src/experiments/configs/base.yaml

# Deactivate the conda environment
conda deactivate