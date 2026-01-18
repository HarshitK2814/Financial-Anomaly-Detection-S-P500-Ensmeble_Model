import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchEncoder(nn.Module):
    """
    Dual-Branch Encoder (Matched to Checkpoint Architecture)
    Using Linear layers for both branches based on weight shapes.
    """
    def __init__(self, seq_len, n_features, hidden_dim=64, latent_dim=32):
        super(DualBranchEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.flat_dim = seq_len * n_features
        
        # Global Branch - MLP on flattened input
        # encoder.global_encoder.0.weight [64, 1280]
        self.global_encoder = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Local Branch - Point-wise MLP (shared across time steps)
        # encoder.local_encoder.0.weight [64, 10] - Inputs are feature dim
        # operates on (Batch, Seq, Features) -> (Batch, Seq, Hidden)
        self.local_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Feature Fusion - Projection Heads
        # encoder.feat_global_fusion.0.weight [64, 32]
        self.feat_global_fusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # encoder.feat_local_fusion.0.weight [64, 32]
        self.feat_local_fusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Global Branch
        x_flat = x.view(batch_size, -1)
        z_global_raw = self.global_encoder(x_flat) # (B, 32)
        
        # Local Branch
        # x is (B, 128, 10), Linear operates on last dim -> (B, 128, 32)
        z_local_seq = self.local_encoder(x)
        
        # Pooling Local Features
        z_local_pooled = torch.mean(z_local_seq, dim=1) # (B, 32) (Mean over time dimension)
        
        # Projections
        feat_global = self.feat_global_fusion(z_global_raw)
        feat_local = self.feat_local_fusion(z_local_pooled)
        
        return z_global_raw, z_local_pooled, feat_global, feat_local

class ContrastiveModule(nn.Module):
    """
    Contrastive Anomaly Detection Module (Matched to Checkpoint)
    """
    def __init__(self, seq_len=128, n_features=10, hidden_dim=64, latent_dim=32, temperature=0.07):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.encoder = DualBranchEncoder(seq_len, n_features, hidden_dim, latent_dim)
        
        # Decoder
        # decoder.0.weight [128, 64]
        # decoder.2.weight [1280, 128]
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.seq_len * self.n_features)
        )

    def encode(self, x):
        # Used by inference
        z_global, z_local, _, _ = self.encoder(x)
        z = torch.cat([z_global, z_local], dim=-1) # (B, 64)
        return z, torch.zeros_like(z)

    def compute_anomaly_score(self, x, normal_prototypes=None, beta=1.0):
        """
        Compute anomaly score
        """
        self.eval()
        with torch.no_grad():
            z_global, z_local, _, _ = self.encoder(x)
            z_combined = torch.cat([z_global, z_local], dim=-1) # (B, 64)
            
            if normal_prototypes is None:
                normal_prototypes = z_combined.mean(dim=0, keepdim=True)
            
            # Simple Euclidean distance to prototype
            dist = torch.cdist(z_combined, normal_prototypes).squeeze(-1)
            
            if len(dist.shape) > 1 and dist.shape[1] > 1:
                dist = dist.min(dim=1)[0]
                
            return dist
