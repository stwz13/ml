import numpy as np
from scipy.integrate import tplquad
from sklearn.metrics import recall_score


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2))

    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    return matrix

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fp = cm[0, 1]

    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fn = cm[1, 0]

    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return np.trace(cm) / np.sum(cm)