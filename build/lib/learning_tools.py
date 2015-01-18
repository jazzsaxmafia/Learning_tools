import pandas as pd
import numpy as np
from gradientDescent import *
from utility import *
from ggplot import *
import sys

def linearRegression(fileName, alpha, iteration):

    data = pd.read_table(fileName, sep=',', header=None, index_col=None)

    m = data.shape[0]
    n = data.shape[1] - 1

    X, mu, sigma = featureNormalize(data[range(n)])
    X['ones'] = 1

    X = np.array(X[['ones']+range(n)])
    y = np.array(data[[n]]).flatten()
    theta = np.zeros(n+1)

    return linear(X, y, theta, alpha, iteration)

def logisticRegression(fileName, alpha, iteration):

    data = pd.read_table(fileName, sep=',', header=None, index_col=None)

    m = data.shape[0]
    n = data.shape[1] - 1

    X, mu, sigma = featureNormalize(data[range(n)])
    X['ones'] = 1

    X = np.array(X[['ones']+range(n)])
    y = np.array(data[[n]]).flatten()
    theta = np.zeros(n+1)

    return logistic(X, y, theta, alpha, iteration)


