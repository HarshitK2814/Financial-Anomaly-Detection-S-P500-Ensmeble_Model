#!/usr/bin/env python
"""
SOTA Anomaly Detection Training Script
=======================================
Trains ConvVAE, Anomaly Transformer, or Contrastive models with optional
regime-aware anomaly labels.

Usage:
    python src/training/train_sota.py \
        --model_type conv_vae \
        --train_data artifacts/market_windows_10f.npy \
        --output_path artifacts/convvae_sota.pt \
        --use_anomaly  # Optional: use regime-aware anomaly labels
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_and_preprocess_data

# ContrastiveModule may have additional dependencies; import optionally
try:
    from src.models.contrastive_module import ContrastiveModule
    CONTRASTIVE_AVAILABLE = True
except ImportError:
    ContrastiveModule = None
    CONTRASTIVE_AVAILABLE = False


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for time series anomaly detection."""

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        enc_out_dim = hidden_dim * 2 * seq_len
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, enc_out_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden_dim * 2, seq_len)),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, n_features, kernel_size=3, padding=1),
        )

    def encode(self, x):
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        out = self.decoder(h.view(h.size(0), -1, self.seq_len))
        return out.permute(0, 2, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_loss(self, x, beta: float = 1.0):
        recon, mu, logvar = self(x)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class AnomalyTransformer(nn.Module):
    """Transformer-based model for anomaly detection."""

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64,
                 latent_dim: int = 32, n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_proj(h)

    def compute_loss(self, x, beta: float = 1.0):
        recon = self(x)
        return nn.functional.mse_loss(recon, x), None, None


def get_model(model_type: str, seq_len: int, n_features: int, args):
    """Factory function to create model based on type."""
    if model_type == 'conv_vae':
        return ConvVAE(seq_len, n_features, args.hidden_dim, args.latent_dim)
    elif model_type == 'anomaly_transformer':
        return AnomalyTransformer(seq_len, n_features, args.hidden_dim,
                                  args.latent_dim, args.n_heads, args.n_layers)
    elif model_type == 'contrastive':
        if not CONTRASTIVE_AVAILABLE:
            raise ImportError("ContrastiveModule not available. Install required dependencies.")
        return ContrastiveModule(seq_len, n_features, args.hidden_dim, args.latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    """Main training loop."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    if args.use_anomaly:
        print("Loading data with regime-aware anomaly labels...")
        data, scaler, regimes, thresholds = load_and_preprocess_data(
            args.train_data,
            window_size=128,
            normalize=True,
            quantile=args.anomaly_quantile,
        )
        print(f"Regime thresholds: {thresholds}")
        print(f"Anomaly rate: {data[:, :, -1].mean():.4f}")
        # Last channel is anomaly label; use for auxiliary supervision if desired
        n_features = data.shape[-1]  # Includes anomaly flag
    else:
        print("Loading data (standard mode, no anomaly labels)...")
        data = np.load(args.train_data)
        n_features = data.shape[-1]

    seq_len = data.shape[1]
    print(f"Data shape: {data.shape} (samples, seq_len, features)")

    # Create DataLoader
    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = get_model(args.model_type, seq_len, n_features, args)
    model = model.to(device)
    print(f"Model: {args.model_type} with {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            if args.model_type == 'contrastive':
                # Contrastive uses its own loss
                loss = model.compute_contrastive_loss(x)
            else:
                loss, _, _ = model.compute_loss(x, beta=args.beta)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'use_anomaly': args.use_anomaly,
            }, args.output_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SOTA anomaly detection models")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['conv_vae', 'anomaly_transformer', 'contrastive'])
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')

    # Regime-aware anomaly label options
    parser.add_argument('--use_anomaly', action='store_true',
                        help='Use regime-aware anomaly labels from loader')
    parser.add_argument('--anomaly_quantile', type=float, default=0.99,
                        help='Quantile for conditional tail threshold (default: 0.99)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
