import numpy as np
from utility import *

def linear(X, y, theta, alpha, iteration):
    for i in range(iteration):
        m = float(len(y))
        hx = np.dot(X, theta)
        result = ((hx - y) * X.T).sum(1)

        theta = theta - alpha*(1/m)*result
        error = 1 / (2*m) * np.square(np.dot(X, theta) - y).sum()

    print "error: ",error
    return theta

def logistic(X, y, theta, alpha, iteration):
    for i in range(iteration):
        m = float(len(y))
        hx = sigmoid(np.dot(X, theta))
        result = ((hx - y) * X.T).sum(1)

        theta = theta - alpha * (1/m) * result
        error = (1/m) * (-1.0*y*np.log(hx) - (1.0 - y)*np.log(1.0 - hx)).sum()


    print "error: ",error
    return theta
