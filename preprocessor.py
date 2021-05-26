import numpy as np


def preprocess_mfcc(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

def preprocess_mel(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X
