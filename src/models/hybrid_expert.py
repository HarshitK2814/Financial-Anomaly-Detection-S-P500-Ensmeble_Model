"""
🔗 HYBRID FACTOR+SEQUENCE EXPERT
===================================
Novel dual-branch architecture that processes BOTH:
  • 32 tabular factors (interpretable branch)
  • 128×10 raw sequences (temporal branch)

Cross-attention allows the model to "explain" which branch dominates
per prediction, providing intrinsic interpretability.

This addresses the key limitation: factors discard sequential info,
while sequences discard interpretability. The hybrid captures both.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HybridExpert(nn.Module):
    """Dual-branch hybrid expert combining tabular factors and sequences.

    Branch 1 (Factor): MLP over 32 interpretable factors
    Branch 2 (Sequence): Lightweight Transformer over raw 128×10 sequence
    Fusion: Cross-attention between branch outputs + learned gating

    Parameters
    ----------
    n_factors : int
        Number of tabular factor features.
    n_features : int
        Number of per-timestep features in raw sequence.
    seq_len : int
        Sequence length.
    d_model : int
        Internal model dimension.
    n_heads : int
        Attention heads for sequence branch and cross-attention.
    dropout : float
        Dropout rate.
    """

    def __init__(self, n_factors: int = 32,
                 n_features: int = 10,
                 seq_len: int = 128,
                 d_model: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        self.d_model = d_model

        # ── Factor Branch (MLP) ──
        self.factor_branch = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── Sequence Branch (Lightweight Transformer) ──
        self.seq_proj = nn.Linear(n_features, d_model)
        self.seq_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Cross-Attention Fusion ──
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # ── Branch Gating (learned) ──
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),  # 2 branches
            nn.Softmax(dim=-1),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # XAI: store gate weights and cross-attention
        self.gate_weights = None
        self.cross_attn_weights = None

    def forward(self, factors: torch.Tensor,
                sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        factors : torch.Tensor, shape (B, D_factors)
            Tabular factor features.
        sequence : torch.Tensor, shape (B, T, F)
            Raw time series sequence.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Anomaly logits.
        """
        # Factor branch
        factor_repr = self.factor_branch(factors)  # (B, d_model)

        # Sequence branch
        seq_repr = self.seq_proj(sequence)  # (B, T, d_model)
        seq_repr = seq_repr + self.seq_pos[:, :sequence.size(1), :]
        seq_repr = self.seq_transformer(seq_repr)  # (B, T, d_model)
        seq_pooled = seq_repr.mean(dim=1)  # (B, d_model)

        # Cross-attention: factor queries sequence context
        factor_query = factor_repr.unsqueeze(1)  # (B, 1, d_model)
        cross_out, self.cross_attn_weights = self.cross_attention(
            factor_query, seq_repr, seq_repr
        )  # (B, 1, d_model)
        cross_out = cross_out.squeeze(1)  # (B, d_model)

        # Gating: learn which branch to trust
        combined = torch.cat([factor_repr, seq_pooled], dim=-1)  # (B, 2*d_model)
        self.gate_weights = self.gate(combined)  # (B, 2)

        # Weighted combination
        fused = (self.gate_weights[:, 0:1] * factor_repr +
                 self.gate_weights[:, 1:2] * cross_out)  # (B, d_model)

        return self.classifier(fused)

    def predict_proba(self, factors: torch.Tensor,
                      sequence: torch.Tensor) -> torch.Tensor:
        logits = self.forward(factors, sequence)
        return torch.sigmoid(logits).squeeze(-1)

    def get_branch_contribution(self) -> Optional[np.ndarray]:
        """Return gate weights showing factor vs sequence contribution.

        Returns
        -------
        np.ndarray, shape (B, 2)
            Column 0 = factor branch weight, Column 1 = sequence branch weight.
        """
        if self.gate_weights is not None:
            return self.gate_weights.detach().cpu().numpy()
        return None

    def get_cross_attention_weights(self) -> Optional[np.ndarray]:
        """Return cross-attention weights showing which timesteps
        the factor branch attends to in the sequence.

        Returns
        -------
        np.ndarray, shape (B, 1, T)
        """
        if self.cross_attn_weights is not None:
            return self.cross_attn_weights.detach().cpu().numpy()
        return None


class HybridTrainer:
    """Training wrapper for HybridExpert."""

    def __init__(self, model: HybridExpert,
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
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _focal_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.focal_gamma * ce).mean()

    def fit(self, X_factors_train, X_seq_train, y_train,
            X_factors_val=None, X_seq_val=None, y_val=None,
            epochs=50, batch_size=32, patience=10, verbose=True):

        from torch.utils.data import DataLoader, TensorDataset

        train_ds = TensorDataset(
            torch.from_numpy(X_factors_train).float(),
            torch.from_numpy(X_seq_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_factors_val is not None:
            val_ds = TensorDataset(
                torch.from_numpy(X_factors_val).float(),
                torch.from_numpy(X_seq_val).float(),
                torch.from_numpy(y_val).float(),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        best_val_loss = float('inf')
        best_state = None
        wait = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch in train_loader:
                factors, seq, y = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                logits = self.model(factors, seq).squeeze(-1)
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
                        factors, seq, y = [b.to(self.device) for b in batch]
                        logits = self.model(factors, seq).squeeze(-1)
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
    model = HybridExpert(n_factors=32, n_features=10, seq_len=128)
    factors = torch.randn(4, 32)
    seq = torch.randn(4, 128, 10)

    logits = model(factors, seq)
    print(f"Factors: {factors.shape}, Seq: {seq.shape}")
    print(f"Output: {logits.shape}")
    print(f"Gate weights: {model.get_branch_contribution()}")
    print(f"Cross-attn: {model.get_cross_attention_weights().shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
