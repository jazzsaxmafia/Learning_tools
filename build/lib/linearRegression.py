import pandas as pd
import numpy as np
from gradientDescent import linear
from utility import *
from ggplot import *
import sys

def main():
    if len(sys.argv) < 4:
        print "Usage: ./ex2.py <fileName>"
        sys.exit(1)

    fileName = sys.argv[1]
    alpha = float(sys.argv[2])
    iteration = int(sys.argv[3])

    data = pd.read_table(fileName, sep=',', header=None, index_col=None)

    m = data.shape[0]
    n = data.shape[1] - 1

    X, mu, sigma = featureNormalize(data[range(n)])
    X['ones'] = 1

    X = np.array(X[['ones']+range(n)])
    y = np.array(data[[n]]).flatten()
    theta = np.zeros(n+1)

    theta = linear(X, y, theta, alpha, iteration)

if __name__=="__main__":
    main()
