import numpy as np
import tensorflow as tf
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, folder)
import hmc2


device = '/GPU:1' ###
ds = [60, 80, 100]
alpha = 1
gamma = 0.5
Delta = 0.1 ###
vlaw = 'ones'
sig = lambda x: (x**2 - 1)/np.sqrt(2.0) + (x**3 - 3*x)/6
params = {'step_size': 0.01,
          'num_leapfrog_steps': 10,
          'num_adaptation_steps': 4000,
          'num_post_adapt_steps': 0}

def main():
    with open('data_2.csv', 'a', newline='') as file: ###
        writer = csv.writer(file)
        for d in ds:
            k = int(gamma*d)
            for i in range(10): ###
                with tf.device(device):
                    W0, X, Y, v = hmc2.data_generate(d, gamma, alpha, Delta, sig, vlaw)
                    W_ = tf.random.normal((k, d), dtype=tf.float32)
                    Ws_unfo = hmc2.hmc(params, W_, X, Y, v, gamma, alpha, Delta, sig, show_acceptance_rate=False, show_adaptation_steps=True)
                    q2s = hmc2.q2(Ws_unfo, W0, v)
                    writer.writerow(np.insert(q2s,0,d))
                    file.flush()

if __name__ == "__main__":
    main()