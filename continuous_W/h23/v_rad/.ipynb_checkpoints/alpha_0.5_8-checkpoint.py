import numpy as np
import tensorflow as tf
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, folder)
import hmc2


def main():
    device='/GPU:1' ###
    d = 150
    gamma = 0.5
    k = int(gamma*d)
    Delta = 1.25 ###
    vlaw = 'rad' ###
    sig = lambda x: (x**2-1)/np.sqrt(2) + (x**3 - 3*x)/6 ###
    ntest = 10000

    alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  ###
    with open('alpha_0.5_8_info.csv', 'a', newline='') as file1, open('alpha_0.5_8_unfo.csv', 'a', newline='') as file2: ###
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        for i in range(4): ###
            mmse1s, mmse2s = [], []
            for alpha in alphas:
                params = {'step_size': 0.01,
                          'num_leapfrog_steps': 10,
                          'num_adaptation_steps': 1900,
                          'num_post_adapt_steps': 100}
                with tf.device(device):
                    W0, X, Y, v = hmc2.data_generate(d, gamma, alpha, Delta, sig, vlaw)
                    W_ = tf.random.normal((k, d), dtype=tf.float32)
                    Ws_unfo = hmc2.hmc(params, W_, X, Y, v, gamma, alpha, Delta, sig, show_acceptance_rate=False, show_adaptation_steps=False)
                    Ws_info = hmc2.hmc(params, W0, X, Y, v, gamma, alpha, Delta, sig, show_acceptance_rate=False, show_adaptation_steps=False)
                    test_info =  hmc2.test_error(Ws_info, W0, v, sig, ntest)
                    test_unfo =  hmc2.test_error(Ws_unfo, W0, v, sig, ntest)
                    mmse1 = tf.reduce_mean(test_info).numpy()
                    mmse2 = tf.reduce_mean(test_unfo).numpy()
                    mmse1s.append(mmse1)
                    mmse2s.append(mmse2)
            writer1.writerow(mmse1s)
            writer2.writerow(mmse2s)
            file1.flush()
            file2.flush()

if __name__ == "__main__":
    main()
