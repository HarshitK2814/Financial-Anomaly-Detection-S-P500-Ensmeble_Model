"""
🧠 LSTM-ATTENTION REGIME EXPERT
=================================
Bidirectional LSTM with multi-head self-attention for sequence-based
anomaly detection within a specific volatility regime.

Key design choices:
  • Bidirectional LSTM captures both forward and backward dependencies
  • Multi-head attention learns which timesteps are most relevant
  • Attention weights are stored for XAI (temporal attribution)
  • Focal loss handles extreme class imbalance (2% anomaly rate)
  • Dropout and weight decay prevent overfitting on small regime subsets
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses learning on hard-to-classify samples (missed anomalies).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class LSTMAttentionExpert(nn.Module):
    """Bidirectional LSTM with Multi-Head Attention for anomaly detection.

    Architecture:
        Input (B, T, F) → BiLSTM → MultiHead Attention → Pool → Classifier

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden_dim : int
        LSTM hidden dimension (each direction).
    n_layers : int
        Number of stacked LSTM layers.
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

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Store attention weights for XAI
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)
            Input sequence.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Anomaly logits (pre-sigmoid).
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)
        lstm_out = self.layer_norm(lstm_out)

        # Self-attention
        attn_out, self.attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (B, T, 2*H), (B, T, T)

        # Weighted average pooling using attention output
        context = attn_out.mean(dim=1)  # (B, 2*H)

        return self.classifier(context)  # (B, 1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict anomaly probability."""
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Return last attention weight matrix for XAI.

        Returns
        -------
        np.ndarray, shape (B, T, T) or None
        """
        if self.attention_weights is not None:
            return self.attention_weights.detach().cpu().numpy()
        return None

    def get_temporal_importance(self, x: torch.Tensor) -> np.ndarray:
        """Compute temporal importance scores from attention weights.

        Returns aggregated attention per timestep, showing which
        time points in the window the model focuses on.

        Returns
        -------
        np.ndarray, shape (B, T)
            Per-timestep importance scores.
        """
        _ = self.forward(x)
        if self.attention_weights is None:
            return np.zeros((x.shape[0], x.shape[1]))

        # Average attention across heads → (B, T, T)
        # Sum over query dimension → (B, T) = "how much attention each
        # key position receives"
        attn = self.attention_weights.detach().cpu().numpy()
        temporal_importance = attn.mean(axis=1)  # Average over queries
        return temporal_importance


class LSTMAttentionTrainer:
    """Training wrapper for LSTMAttentionExpert with focal loss.

    Parameters
    ----------
    model : LSTMAttentionExpert
        The model to train.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    focal_alpha : float
        Focal loss alpha (class weight).
    focal_gamma : float
        Focal loss gamma (focusing parameter).
    device : str
        Device to train on.
    """

    def __init__(self, model: LSTMAttentionExpert,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, min_lr=1e-6
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(x).squeeze(-1)
            loss = self.criterion(logits, y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate the model. Returns dict with loss and predictions."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device).float()

                logits = self.model(x).squeeze(-1)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.cpu().numpy())

        return {
            'loss': total_loss / len(dataloader),
            'probs': np.array(all_probs),
            'labels': np.array(all_labels),
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32,
            patience: int = 10, verbose: bool = True) -> dict:
        """Full training loop with early stopping.

        Returns training history dict.
        """
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float()
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            if val_loader:
                val_res = self.evaluate(val_loader)
                val_loss = val_res['loss']
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
                    print(f"  Epoch {epoch+1}/{epochs} | "
                          f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
