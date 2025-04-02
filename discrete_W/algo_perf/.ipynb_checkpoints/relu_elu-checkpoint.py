import numpy as np
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, folder)

import algo

def main():
    d = 150
    gamma = 0.5
    Delta = 0.1
    k = int(gamma*d)
    v = np.ones(k)
    phi1 = lambda x: np.maximum(x,0) # ReLU
    phi2 = lambda x: np.where(x > 0, x, (np.exp(x) - 1)) # ELU 
    
    alphas = np.linspace(0.2, 4, 20)
    with open(f'relu.csv', 'a', newline='') as file1, open(f'elu.csv', 'a', newline='') as file2:
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        for i in range(16):
            mmse1 = [algo.algo_perf(d, gamma, alpha, Delta, phi1, v) for alpha in alphas]
            writer1.writerow(mmse1)
            file1.flush()
            
            mmse2 = [algo.algo_perf(d, gamma, alpha, Delta, phi2, v) for alpha in alphas]
            writer2.writerow(mmse2)
            file2.flush()

if __name__ == "__main__": 
    main()