"""
Contrastive Anomaly Detection Module
Inspired by DCdetector (KDD 2023) and CARLA (Pattern Recognition 2024)
Key: Dual-branch contrastive representation without reconstruction
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import sys

# Add src directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# FIXED IMPORT - use relative import
from .dual_branch_encoder import DualBranchEncoder

class ContrastiveModule(nn.Module):
    """
    Contrastive Anomaly Detection
    Normal samples cluster tightly, anomalies are far from cluster centers
    """
    def __init__(self, seq_len=128, n_features=10, hidden_dim=64, latent_dim=32, temperature=0.07):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.encoder = DualBranchEncoder(seq_len, n_features, hidden_dim, latent_dim)
        self.fc_mu = nn.Identity()
        self.fc_logvar = nn.Identity()

        # DECLARE DECODER HERE so state_dict works
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.seq_len * self.n_features)
        )

    def encode(self, x):
        z_global, z_local, _, _ = self.encoder(x)
        z = torch.cat([z_global, z_local], dim=-1)
        return z, torch.zeros_like(z)  # dummy logvar for compatibility

    def forward(self, x):
        """FIXED forward method with proper dimension handling"""
        z_global, z_local, feat_global, feat_local = self.encoder(x)
        
        # FIXED: Pool the local representations to match global dimensions
        # Method: Average pooling over sequence dimension
        z_local_pooled = torch.mean(z_local, dim=1)  # (batch_size, latent_dim)
        
        # Concatenate global and pooled local representations
        z_combined = torch.cat([z_global, z_local_pooled], dim=-1)  # (batch_size, 2 * latent_dim)
        
        recon = self.decoder(z_combined).view(x.shape)
        mu = z_combined
        logvar = torch.zeros_like(mu)
        return recon, mu, logvar
    
    def encode_samples(self, x):
        """Encode samples for contrastive learning - FIXED to handle dimension mismatch"""
        z_global, z_local, _, _ = self.encoder(x)
        
        # Pool the local representations to match global dimensions
        # Method 1: Average pooling over sequence dimension
        z_local_pooled = torch.mean(z_local, dim=1)  # (batch_size, latent_dim)
        
        # Concatenate global and pooled local representations
        z_combined = torch.cat([z_global, z_local_pooled], dim=-1)  # (batch_size, 2 * latent_dim)
        return z_combined

    def compute_anomaly_score(self, x, normal_prototypes=None, beta=1.0):
        """
        Compute anomaly score as distance from normal cluster center
        If normal_prototypes not provided, use batch mean as proxy
        """
        self.eval()
        with torch.no_grad():
            z_combined = self.encode_samples(x)
            if normal_prototypes is None:
                normal_prototypes = z_combined.mean(dim=0, keepdim=True)  # Use batch mean
            dist = torch.cdist(z_combined, normal_prototypes).squeeze(-1)
            return dist

class ContrastiveLoss(nn.Module):
    """Contrastive Loss"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.relu(dist_pos - dist_neg + self.margin))
        return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModule(seq_len=128, n_features=10, latent_dim=32).to(device)
    model.eval()
    x = torch.randn(8, 128, 10, device=device)
    with torch.no_grad():
        scores = model.compute_anomaly_score(x)
        print(f"Input: {x.shape}")
        print(f"Scores: {scores.shape}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
