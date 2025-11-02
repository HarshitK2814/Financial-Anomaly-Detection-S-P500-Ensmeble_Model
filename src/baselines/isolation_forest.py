from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestBaseline:
    def __init__(self, contamination=0.1, random_state=None):
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.score_samples(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def get_anomaly_scores(self, X):
        return -self.score_samples(X)  # Higher scores indicate anomalies

# Example usage:
# if __name__ == "__main__":
#     # Load your data here
#     data = np.random.rand(100, 10)  # Replace with actual data
#     model = IsolationForestBaseline(contamination=0.05)
#     model.fit(data)
#     predictions = model.predict(data)
#     anomaly_scores = model.get_anomaly_scores(data)
#     print(predictions)
#     print(anomaly_scores)