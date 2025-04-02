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
    sig = lambda x: np.maximum(x,0)
    
    alphas = [0.8, 1.0, 1.2] ###
    ncycles = [300, 500, 500] ###
    Ts = [800, 800, 800] ###
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(16):
            for alpha, ncycle, T in zip(alphas, ncycles, Ts):
                W0, X, Y = data_generate(d, gamma, alpha, Delta, v, sig)
                W1s = metropolis(W0, X, Y, Delta, v, sig, ncycle, info=False)
                W2s = metropolis(W0, X, Y, Delta, v, sig, ncycle, info=False)
                qw = [np.sum(W0*W1)/(k*d) for W1 in W1s]
                qw = np.mean(qw[T:])
                row = [alpha, qw]
                for l in [1,2,3,4,5,6]:
                    q00 = v.T@np.power(W0@W0.T, l)@v/(k*d**l)
                    q01 = np.array([v.T@np.power(W0@W1.T, l)@v/(k*d**l) for W1 in W1s])
                    q12 = np.array([v.T@np.power(W1@W2.T, l)@v/(k*d**l) for W1, W2 in zip(W1s, W2s)])
                    err = q00 - 2*q01 + q12
                    err = np.mean(err[T:])
                    row.append(err)
                writer.writerow(row)
                file.flush()

if __name__ == "__main__":
    main()
