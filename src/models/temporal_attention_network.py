"""
⚡ REGIME-AWARE TEMPORAL ATTENTION NETWORK (RA-TAN)
=====================================================
NOVEL ARCHITECTURE CONTRIBUTION

A Transformer-based architecture that injects regime embeddings into
the attention mechanism, so the model learns regime-specific temporal
patterns for anomaly detection.

Innovation over standard Anomaly Transformers:
  • Regime embedding is added to every position embedding, conditioning
    attention patterns on the current volatility regime.
  • This means the model learns that "what matters temporally" changes
    across regimes:
      - Low Vol → micro-liquidity events, flash crashes
      - Med Vol → momentum decay, trend reversals
      - High Vol → panic cascades, volatility clustering

Key XAI features:
  • Multi-head attention maps per regime (visualizable)
  • Regime-conditioned saliency scores
  • Attention entropy as an anomaly indicator
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RegimeAwareTemporalAttention(nn.Module):
    """Regime-Aware Temporal Attention Network (RA-TAN).

    Novel architecture: Injects regime embeddings into the Transformer
    attention mechanism so the model learns regime-specific temporal
    patterns for anomaly detection.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    d_model : int
        Transformer model dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    n_regimes : int
        Number of volatility regimes.
    seq_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_features: int = 10,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 n_regimes: int = 3,
                 seq_len: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Regime embedding — learns a vector per regime
        self.regime_embedding = nn.Embedding(n_regimes, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder with custom layers for attention extraction
        self.layers = nn.ModuleList([
            TransformerLayerWithAttention(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Store attention maps for XAI
        self.all_attention_weights = []

    def forward(self, x: torch.Tensor,
                regime_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with regime conditioning.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)
            Input time series.
        regime_ids : torch.Tensor, shape (B,)
            Integer regime labels (0=Low, 1=Med, 2=High).

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Anomaly logits.
        """
        B, T, F = x.shape

        # Project input features to model dimension
        h = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        h = self.pos_encoding(h)

        # ★ NOVEL: Inject regime context into every timestep
        regime_emb = self.regime_embedding(regime_ids)  # (B, d_model)
        regime_emb = regime_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)
        h = h + regime_emb  # Regime-aware representation

        # Pass through Transformer layers, collecting attention weights
        self.all_attention_weights = []
        for layer in self.layers:
            h, attn_weights = layer(h)
            self.all_attention_weights.append(attn_weights)

        # Pool: mean over time
        context = h.mean(dim=1)  # (B, d_model)

        return self.classifier(context)

    def predict_proba(self, x: torch.Tensor,
                      regime_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, regime_ids)
        return torch.sigmoid(logits).squeeze(-1)

    def get_attention_maps(self, layer_idx: int = -1) -> Optional[np.ndarray]:
        """Return attention weights from a specific layer.

        Parameters
        ----------
        layer_idx : int
            Which layer's attention to return. -1 for last.

        Returns
        -------
        np.ndarray, shape (B, n_heads, T, T) or None
        """
        if not self.all_attention_weights:
            return None
        return self.all_attention_weights[layer_idx].detach().cpu().numpy()

    def get_regime_attention_comparison(self) -> dict:
        """Compare attention patterns across regimes (post-inference).

        Useful for XAI: shows how attention shifts between regimes.
        """
        # This should be called after batch inference with known regime labels
        if not self.all_attention_weights:
            return {}

        last_attn = self.all_attention_weights[-1].detach().cpu().numpy()
        # Compute attention entropy per sample
        # Higher entropy = more distributed attention = less focused
        entropy = -np.sum(
            last_attn * np.log(last_attn + 1e-8), axis=-1
        ).mean(axis=1)  # (B,)

        return {
            'attention_entropy': entropy,
            'mean_entropy': float(entropy.mean()),
        }


class TransformerLayerWithAttention(nn.Module):
    """Transformer encoder layer that exposes attention weights."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class RATANTrainer:
    """Training wrapper for RA-TAN with focal loss.

    Parameters
    ----------
    model : RegimeAwareTemporalAttention
    lr : float
    weight_decay : float
    device : str
    """

    def __init__(self, model: RegimeAwareTemporalAttention,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-4,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _focal_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.focal_gamma * ce).mean()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            regimes_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            regimes_val: np.ndarray = None,
            epochs: int = 60, batch_size: int = 32,
            patience: int = 15, verbose: bool = True) -> dict:
        """Full training loop.

        Parameters
        ----------
        X_train : (N, T, F) sequence data
        y_train : (N,) binary labels
        regimes_train : (N,) regime labels
        """
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
            torch.from_numpy(regimes_train).long(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
                torch.from_numpy(regimes_val).long(),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch in train_loader:
                x, y, r = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                logits = self.model(x, r).squeeze(-1)
                loss = self._focal_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y, r = [b.to(self.device) for b in batch]
                        logits = self.model(x, r).squeeze(-1)
                        val_loss += self._focal_loss(logits, y).item()
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in
                                  self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"  Epoch {epoch+1}/{epochs} | Train: {avg_loss:.4f}"
                if val_loader:
                    msg += f" | Val: {val_loss:.4f}"
                print(msg)

        if best_state:
            self.model.load_state_dict(best_state)
        return history


if __name__ == '__main__':
    # Quick test
    device = 'cpu'
    model = RegimeAwareTemporalAttention(
        n_features=10, d_model=64, n_heads=4, n_layers=3
    ).to(device)

    x = torch.randn(4, 128, 10)
    regimes = torch.tensor([0, 1, 2, 1])

    logits = model(x, regimes)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Attention maps: {len(model.all_attention_weights)} layers")
    print(f"  Layer 0 shape: {model.all_attention_weights[0].shape}")

    probs = model.predict_proba(x, regimes)
    print(f"Probabilities: {probs}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
