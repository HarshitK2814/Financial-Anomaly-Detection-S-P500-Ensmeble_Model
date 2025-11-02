import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ART = os.path.join(ROOT, 'artifacts')
os.makedirs(ART, exist_ok=True)

# match shapes used by your ConvVAE smoke run
N, seq_len, n_features = 1024, 128, 10
rng = np.random.RandomState(42)
X = rng.randn(N, seq_len, n_features).astype(np.float32)

out = os.path.join(ART, 'multivariate_windows.npy')
np.save(out, X)
print("Saved", out, "shape=", X.shape)