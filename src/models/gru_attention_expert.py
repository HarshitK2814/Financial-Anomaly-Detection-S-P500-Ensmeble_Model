"""
🧠 GRU-ATTENTION REGIME EXPERT
================================
Gated Recurrent Unit with attention for sequence-based anomaly detection.

Lighter alternative to LSTM-Attention:
  • Fewer parameters (2 gates vs 3)
  • Faster training on small per-regime datasets
  • Comparable performance on short-to-medium sequences
  • Same attention-based XAI interface
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class GRUAttentionExpert(nn.Module):
    """Bidirectional GRU with Multi-Head Attention.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_dim : int
        GRU hidden dimension per direction.
    n_layers : int
        Number of stacked GRU layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_features: int = 10,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)

        attn_out, self.attention_weights = self.attention(
            gru_out, gru_out, gru_out
        )

        context = attn_out.mean(dim=1)
        return self.classifier(context)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    def get_attention_weights(self) -> Optional[np.ndarray]:
        if self.attention_weights is not None:
            return self.attention_weights.detach().cpu().numpy()
        return None

    def get_temporal_importance(self, x: torch.Tensor) -> np.ndarray:
        _ = self.forward(x)
        if self.attention_weights is None:
            return np.zeros((x.shape[0], x.shape[1]))
        attn = self.attention_weights.detach().cpu().numpy()
        return attn.mean(axis=1)


class GRUAttentionTrainer:
    """Training wrapper with focal loss and early stopping."""

    def __init__(self, model: GRUAttentionExpert,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, min_lr=1e-6
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
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32,
            patience: int = 10, verbose: bool = True) -> dict:

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float()
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
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x).squeeze(-1)
                loss = self._focal_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                        logits = self.model(x).squeeze(-1)
                        val_loss += self._focal_loss(logits, y).item()
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

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
