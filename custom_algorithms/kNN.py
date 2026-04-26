from collections import Counter

import numpy as np


class kNN:
    def __init__(self, k=5, metric="euclidian"):
        self.k = k
        self.metric = metric

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_one_obj(x) for x in X])

    def predict_one_obj(self, x):
        distances = self.find_distance(x, self.X_train)
        k_nearest_obj = np.argpartition(distances, self.k)[:self.k]

        k_nearest_obj_labels = self.y_train[k_nearest_obj]

        return Counter(k_nearest_obj_labels).most_common(1)[0][0]

    def find_distance(self, x, X):
        if self.metric == "euclidian":
            return np.sqrt(np.sum((x - X) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(x - X), axis=1)
        else:
            raise Exception("There are no another metrics yet")