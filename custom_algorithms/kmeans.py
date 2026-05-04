import numpy as np


class kmeans:
    def __init__(self, k=2, random_state = 42, max_iter=100):
        self.k = k
        self.random_state = random_state
        self.max_iter = max_iter

        self.centroids = None
        self.inertia = None

    def fit(self, X):
        rs = np.random.RandomState(self.random_state)
        self.centroids = X[rs.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            self.centroids = new_centroids


        final_dist = np.linalg.norm(X - self.centroids[labels], axis=1)
        self.inertia = np.sum(final_dist**2)

        return labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


