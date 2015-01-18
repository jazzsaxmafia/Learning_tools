import numpy as np
def gradientDescent(X, y, theta, alpha, iteration):
    for i in range(iteration):
        m = float(len(y))
        hx = np.dot(X, theta)
        result = ((hx - y) * X.T).sum(1)

        theta = theta - alpha*(1/m)*result
        error = 1 / (2*m) * np.square(np.dot(X, theta) - y).sum()

        print error
    return theta

