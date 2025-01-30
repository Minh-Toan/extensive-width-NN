import numpy as np
import theory_universal as uni
from amp import *


def algo_perf(d, gamma, alpha, Delta, phi, v):
    '''
    generate one dataset, perform the algorithm on this dataset, and compute the generalization error
    '''
    c0, c1, c2, nu = uni.coeffs(phi)
    # generate data
    k = int(gamma*d)
    n = int(alpha*d**2)
    W0 = np.random.choice([-1,1],(k,d))
    Dv = np.diag(v)
    S0 = W0.T@Dv@W0
    X = np.random.normal(size=(d,n))
    Z = np.random.randn(n)
    U0 = W0@X/np.sqrt(d)
    Y = phi(U0).T@v/np.sqrt(k) + np.sqrt(Delta)*Z

    # estimate S1 = v.T W/ sqrt(k) by MCMC
    ncycle = 10
    W = np.random.choice([-1,1],(k,d))
    if abs(c1)>1e-6:
        U = W@X/np.sqrt(d)
        D = phi(U).T@v/np.sqrt(k) - Y
        H = np.sum(D**2)/(2*Delta)
        for iteration in range(ncycle*k*d):
            s, t = np.random.randint(0,k), np.random.randint(0, d)
            dU = -2*W[s,t]*X[t]/np.sqrt(d)
            dD =  v[s]*(phi(U[s]+dU) - phi(U[s]))/np.sqrt(k)
            Hnew = np.sum((D+dD)**2)/(2*Delta)
            dH = Hnew - H
            if dH<0 or np.random.rand()<np.exp(-dH): # acceptance condition
                U[s] += dU
                D += dD
                H = Hnew
                W[s,t] *= -1
    S1_hat = v.T@W/np.sqrt(k)
    
    # estimate S2 = W.T Dv W / sqrt(k) by AMP
    Y0 = c0*np.sum(v)/np.sqrt(k)
    Y1 = c1*S1_hat@X/np.sqrt(d)
    Y_ = np.sqrt(2)*(Y - Y0 -Y1)/c2
    noise = 2*(Delta+nu-c0**2-c1**2-c2**2)/(c2**2)
    sensors = (np.einsum("im,jm->mij", X, X) - np.einsum("ij,m->mij", np.eye(d), np.ones(n)))
    S2_hat = AMP(sensors, Y_, gamma, noise, iterations = 100, damping = 0.4, tol = 1e-4)

    # test error of the algorithm
    ntest = 1000
    errs = []
    for i in range(ntest):
        xtest = np.random.randn(d)
        ytest = v.T@phi(W0@xtest/np.sqrt(d))/np.sqrt(k)
        Ztest = np.outer(xtest, xtest)-np.identity(d)
        yhat  = c0*np.sum(v)/np.sqrt(k) + c1*S1_hat@xtest/np.sqrt(d) + c2*np.sum(Ztest*S2_hat)/(d*np.sqrt(2))
        err = (ytest-yhat)**2
        errs.append(err)
    return np.mean(errs)


