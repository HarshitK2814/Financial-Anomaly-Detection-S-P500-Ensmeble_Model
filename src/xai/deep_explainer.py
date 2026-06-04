"""
🧠 DEEP LEARNING EXPLAINER
============================
XAI techniques for neural network models (LSTM, GRU, Transformer, Hybrid).

Techniques:
  1. Attention weight extraction and visualization
  2. Integrated Gradients for input→output attribution
  3. Temporal saliency maps
  4. Attention entropy as anomaly confidence indicator
"""
import numpy as np
import torch
from typing import Optional, Dict, List


class DeepExplainer:
    """Explainability wrapper for deep learning anomaly detection models.

    Supports LSTM-Attention, GRU-Attention, RA-TAN, and Hybrid models.

    Parameters
    ----------
    model : nn.Module
        Fitted PyTorch model.
    feature_names : list of str, optional
        Names of per-timestep features.
    device : str
        Device for computation.
    """

    def __init__(self, model, feature_names: list = None,
                 device: str = 'cpu'):
        self.model = model.eval()
        self.feature_names = feature_names
        self.device = torch.device(device)

    def attention_weights(self, x: torch.Tensor,
                          **kwargs) -> Optional[np.ndarray]:
        """Extract attention weights from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        **kwargs
            Additional args (e.g., regime_ids for RA-TAN).

        Returns
        -------
        np.ndarray or None
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            if 'regime_ids' in kwargs:
                _ = self.model(x, kwargs['regime_ids'].to(self.device))
            else:
                _ = self.model(x)

        if hasattr(self.model, 'get_attention_weights'):
            return self.model.get_attention_weights()
        elif hasattr(self.model, 'all_attention_weights'):
            if self.model.all_attention_weights:
                return self.model.all_attention_weights[-1].detach().cpu().numpy()
        return None

    def temporal_importance(self, x: torch.Tensor,
                            **kwargs) -> np.ndarray:
        """Compute per-timestep importance scores.

        Uses attention weights to determine which timesteps
        the model focuses on for its prediction.

        Returns
        -------
        np.ndarray, shape (B, T)
        """
        if hasattr(self.model, 'get_temporal_importance'):
            x = x.to(self.device)
            return self.model.get_temporal_importance(x)

        # Fallback: use attention weights
        attn = self.attention_weights(x, **kwargs)
        if attn is not None:
            # Average across heads and queries → per-key importance
            if attn.ndim == 4:  # (B, heads, T, T)
                return attn.mean(axis=(1, 2))
            elif attn.ndim == 3:  # (B, T, T)
                return attn.mean(axis=1)

        return np.ones((x.shape[0], x.shape[1])) / x.shape[1]

    def integrated_gradients(self, x: torch.Tensor,
                              target_class: int = 1,
                              n_steps: int = 50,
                              **kwargs) -> np.ndarray:
        """Compute Integrated Gradients for input attribution.

        IG attributes the prediction to each input feature by
        integrating gradients along a straight-line path from a
        baseline (zero input) to the actual input.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)
        target_class : int
            Which class to compute attributions for.
        n_steps : int
            Number of interpolation steps.

        Returns
        -------
        np.ndarray, shape (B, T, F)
            Per-feature, per-timestep attribution scores.
        """
        x = x.to(self.device).requires_grad_(False)
        baseline = torch.zeros_like(x)

        # Accumulate gradients along the path
        integrated_grads = torch.zeros_like(x)

        for step in range(n_steps + 1):
            alpha = step / n_steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            # Forward pass
            if 'regime_ids' in kwargs:
                output = self.model(interpolated, kwargs['regime_ids'].to(self.device))
            else:
                output = self.model(interpolated)

            # Use sigmoid output for binary classification
            if output.shape[-1] == 1:
                score = torch.sigmoid(output).sum()
            else:
                score = output[:, target_class].sum()

            score.backward()
            integrated_grads += interpolated.grad.detach()

        # Scale by input difference and average over steps
        attributions = (x - baseline).detach() * integrated_grads / (n_steps + 1)
        return attributions.cpu().numpy()

    def attention_entropy(self, x: torch.Tensor,
                           **kwargs) -> np.ndarray:
        """Compute attention entropy as an anomaly confidence indicator.

        Low entropy → model is focused → confident prediction
        High entropy → model is uncertain → distributed attention

        Returns
        -------
        np.ndarray, shape (B,)
            Attention entropy per sample.
        """
        attn = self.attention_weights(x, **kwargs)
        if attn is None:
            return np.zeros(x.shape[0])

        # Handle different attention shapes
        if attn.ndim == 4:  # (B, heads, T, T)
            attn_probs = attn.mean(axis=1)  # Average over heads
        elif attn.ndim == 3:
            attn_probs = attn
        else:
            return np.zeros(x.shape[0])

        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(
            attn_probs * np.log(attn_probs + 1e-10), axis=-1
        ).mean(axis=-1)  # (B,)

        return entropy

    def feature_saliency(self, x: torch.Tensor,
                          **kwargs) -> np.ndarray:
        """Compute per-feature saliency via gradient magnitude.

        Simpler than Integrated Gradients but faster.

        Returns
        -------
        np.ndarray, shape (B, F)
            Aggregated per-feature importance.
        """
        x = x.to(self.device).requires_grad_(True)

        if 'regime_ids' in kwargs:
            output = self.model(x, kwargs['regime_ids'].to(self.device))
        else:
            output = self.model(x)

        score = torch.sigmoid(output).sum()
        score.backward()

        grads = x.grad.detach().cpu().numpy()
        # Aggregate: mean absolute gradient over time
        return np.abs(grads).mean(axis=1)  # (B, F)

    def explain_prediction(self, x: torch.Tensor,
                            sample_idx: int = 0,
                            **kwargs) -> dict:
        """Generate comprehensive explanation for a single prediction.

        Returns
        -------
        dict with temporal importance, top features, attention entropy.
        """
        x_batch = x[sample_idx:sample_idx+1]

        # Temporal importance
        temp_imp = self.temporal_importance(x_batch, **kwargs)[0]
        top_timesteps = np.argsort(temp_imp)[::-1][:5]

        # Feature saliency
        feat_sal = self.feature_saliency(x_batch, **kwargs)[0]
        top_features_idx = np.argsort(feat_sal)[::-1][:5]

        top_features = []
        for idx in top_features_idx:
            fname = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else f'Feature_{idx}'
            top_features.append({
                'feature': fname,
                'saliency': float(feat_sal[idx]),
            })

        # Attention entropy
        entropy = self.attention_entropy(x_batch, **kwargs)

        return {
            'sample_index': sample_idx,
            'top_timesteps': top_timesteps.tolist(),
            'temporal_importance': temp_imp.tolist(),
            'top_features': top_features,
            'attention_entropy': float(entropy[0]),
            'confidence': 'high' if entropy[0] < 2.0 else 'moderate' if entropy[0] < 3.5 else 'low',
        }
