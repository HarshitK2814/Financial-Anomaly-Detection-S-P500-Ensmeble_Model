from sklearn.decomposition import PCA
import numpy as np

class PCABaseline:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def inverse_transform(self, X):
        return self.pca.inverse_transform(X)

    def score_samples(self, X):
        transformed = self.transform(X)
        reconstruction = self.inverse_transform(transformed)
        mse = np.mean((X - reconstruction) ** 2, axis=1)
        return mse

    def fit_predict(self, X):
        self.fit(X)
        return self.score_samples(X)