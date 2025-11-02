#!/bin/bash
# Complete SOTA Anomaly Detection Pipeline
# Trains all models and evaluates with ensemble

set -e  # Exit on error

echo "=========================================="
echo "SOTA Anomaly Detection Pipeline"
echo "=========================================="

# Activate environment
#source .venv/bin/activate  # or: conda activate your_env

# Create artifacts directory
mkdir -p artifacts

# Step 1: Train ConvVAE
echo ""
echo "[1/4] Training ConvVAE..."
python src/training/train_sota.py \
    --model_type conv_vae \
    --train_data artifacts/market_windows_10f.npy \
    --output_path artifacts/convvae_sota.pt \
    --hidden_dim 64 \
    --latent_dim 32 \
    --beta 0.5 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --patience 15 \
    --device cuda

# Step 2: Train Anomaly Transformer
echo ""
echo "[2/4] Training Anomaly Transformer..."
python src/training/train_sota.py \
    --model_type anomaly_transformer \
    --train_data artifacts/market_windows_10f.npy \
    --output_path artifacts/anomaly_transformer.pt \
    --hidden_dim 64 \
    --latent_dim 32 \
    --n_heads 4 \
    --n_layers 3 \
    --beta 1.0 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --patience 15 \
    --device cuda

# Step 3: Train Contrastive Module
echo ""
echo "[3/4] Training Contrastive Module..."
python src/training/train_sota.py \
    --model_type contrastive \
    --train_data artifacts/market_windows_10f.npy \
    --output_path artifacts/contrastive.pt \
    --hidden_dim 64 \
    --latent_dim 32 \
    --epochs 150 \
    --batch_size 64 \
    --lr 1e-4 \
    --patience 20 \
    --device cuda

# Step 4: Ensemble Evaluation
echo ""
echo "[4/4] Ensemble Evaluation..."
python src/evaluation/ensemble_evaluate.py \
    --test_data artifacts/market_windows_10f.npy \
    --labels artifacts/market_labels.npy \
    --conv_vae_path artifacts/convvae_sota.pt \
    --transformer_path artifacts/anomaly_transformer.pt \
    --contrastive_path artifacts/contrastive.pt \
    --use_baselines \
    --ensemble_method mean \
    --tolerance 3 \
    --smooth_sigma 2.0 \
    --device cuda

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "Check artifacts/ for results"
echo "=========================================="
