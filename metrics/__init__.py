import numpy as np

def r_2(y_test, y_predict):
    return 1 - np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

def mae(y_test, y_predict):
    return 1/len(y_test) * np.sum(abs(y_predict - y_test))

def mse(y_test, y_predict):
    return 1 / len(y_test) * np.sum((y_predict - y_test) ** 2)

def rmse(y_test, y_predict):
    return mse(y_predict, y_test) ** 0.5

def mape(y_test, y_predict, eps=1e-8):
    return 1/len(y_test) * np.sum(abs(y_predict - y_test) / (y_test + eps))