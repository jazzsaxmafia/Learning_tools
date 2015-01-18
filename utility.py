import numpy as np

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

