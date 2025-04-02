import numpy as np
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, folder)
from general_NN_MCMC import *



def main():
    d = 150
    gamma = 0.5
    Delta = 0.1
    k = int(gamma*d)
    v = np.ones(k)
    sig = lambda x: np.where(x > 0, x, (np.exp(x) - 1))
    
    alphas = [0.1, 0.2, 0.3, 0.4] ###
    ncycles = [50, 50, 100, 500] ###
    Ts = [500, 500, 500, 800] ###
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(16):
            for alpha, ncycle, T in zip(alphas, ncycles, Ts):
                W0, X, Y = data_generate(d, gamma, alpha, Delta, v, sig)
                Ws = metropolis(W0, X, Y, Delta, v, sig, ncycle, info=False)
                qw = [np.sum(W0*W)/(k*d) for W in Ws]
                qw = np.mean(qw[T:])
                row = [alpha, qw]
                for l in [1,2,3,4,5,6]:
                    q00 = v.T@np.power(W0@W0.T, l)@v/(k*d**l)
                    q01 = np.array([v.T@np.power(W0@W.T, l)@v/(k*d**l) for W in Ws])
                    err = q00 - q01
                    err = np.mean(err[T:])
                    row.append(err)
                writer.writerow(row)
                file.flush()

if __name__ == "__main__":
    main()
