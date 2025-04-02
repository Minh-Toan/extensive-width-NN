import numpy as np
import tensorflow as tf
from scipy.integrate import quad
import csv

import sys
import os
folder = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, folder)
import hmc2_ext as hmc2


def main():
    device='/GPU:1' ###
    with tf.device(device):
        phi_ = lambda x: tf.math.erf(x/tf.sqrt(2.0))
        m1 = quad(lambda z: phi_(z)*np.exp(-z**2/2)/np.sqrt(2*np.pi), -10, 10)[0]
        m2 = quad(lambda z: phi_(z)**2*np.exp(-z**2/2)/np.sqrt(2*np.pi), -10, 10)[0]
        std = np.sqrt(m2-m1**2)
        phi = lambda x: (phi_(x)-m1)/std
        
    d = 150
    gamma = 0.5
    k = int(gamma*d)
    Delta = 1.25 ###
    vlaw = 'ones' ###
    sig = lambda x: (x**2-1)/np.sqrt(2) + (x**3 - 3*x)/6 ###
    ntest = 10000

    params = {'step_size': 0.01,
              'num_leapfrog_steps': 10,
              'num_adaptation_steps': 2000,
              'num_post_adapt_steps': 100}

    alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  ###
    with open('alpha_0.5_8_info.csv', 'a', newline='') as file1, open('alpha_0.5_8_unfo.csv', 'a', newline='') as file2: ###
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        for i in range(4): ###
            mmse1s, mmse2s = [], []
            for alpha in alphas:
                with tf.device(device):
                    U0, X, Y, v = hmc2.data_generate(d, gamma, alpha, Delta, sig, phi, vlaw)
                    Us_unfo = hmc2.hmc(params, U0, X, Y, v, gamma, alpha, Delta, sig, phi, info=False, show_acceptance_rate=False, show_adaptation_steps=False)
                    Us_info = hmc2.hmc(params, U0, X, Y, v, gamma, alpha, Delta, sig, phi, info=True,  show_acceptance_rate=False, show_adaptation_steps=False)
                    test_unfo =  hmc2.test_error(Us_unfo, U0, v, sig, phi, ntest)
                    test_info =  hmc2.test_error(Us_info, U0, v, sig, phi, ntest)
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
