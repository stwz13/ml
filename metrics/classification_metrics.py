import numpy as np
from scipy.integrate import tplquad
from sklearn.metrics import recall_score


def accuracy(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))

    n_classes = len(classes)

    matrix = np.zeros((n_classes, n_classes), dtype = int)

    class_idx = {cls: i for i, cls in enumerate(classes)}

    for t,p in zip(y_true, y_pred):
        matrix[class_idx[t], class_idx[p]] += 1

    return matrix

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    precisions = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i], i) - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    recalls = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    return np.mean(recalls)

def f1(y_true, y_pred):
    p = calculate_precision_macro(y_true, y_pred)
    r = calculate_recall_macro(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0