import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect
import scipy.integrate
from activations import *

def bisect_roots(f, a, b):
    '''
    find roots in [a,b] of a function f
    '''
    x_vals = np.linspace(a, b, 100)
    f_vals = [f(x_val) for x_val in x_vals]
    # Find intervals where the function changes sign
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    # Use bisection to find each root
    roots = []
    for i in sign_changes:
        root = bisect(f, x_vals[i], x_vals[i + 1])
        roots.append(root)
    return roots

def m(l):
    '''
    overlap E[XE[X|Y]] of the Gaussian channel Y = sqrt(l)*X + Z
    X Rademacher
    '''
    integrand = lambda z: np.tanh(np.sqrt(l)*z + l)*np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return quad(integrand, -10, 10)[0]

def m_inv(q):
    '''
    inverse of function m
    '''
    u = 1
    if m(u)>=q:
         r = bisect(lambda x: m(x)-q, 0, 1)
    else:
        while m(u)<q:
            u *= 2
        r = bisect(lambda x: m(x)-q, u/2, u)
    return r

def psi(l):
    '''
    free energy function of the Gaussian channel sqrt(l)*X + Z
    X Rademacher
    '''
    integrand = lambda z: np.log( np.cosh( np.sqrt(l)*z + l ) )*np.exp( -z**2/2 )/np.sqrt(2*np.pi)
    return -l/2 + quad(integrand, -10, 10)[0]



def quantities(activation, alpha, gamma, Delta):
    '''
    compute mmse, qw, q2, free entropy, mutual info and potential function in q_W
    '''
    g, g_prime, coeff = gg(activation)
    
    def q2r2(q_W): # compute q2, r2 in terms of qw
        roots = np.roots([coeff[2]**2/2, 
                          -(Delta + coeff[2]**2/2 + g(1)-g(q_W) + alpha*coeff[2]**2*(1-q_W**2)) - coeff[2]**2*q_W**2/2,
                          q_W**2*(Delta+coeff[2]**2/2 + (g(1)-g(q_W))) + 2*alpha*coeff[2]**2/2*(1-q_W**2)]).real
        mask = (roots>=0)&(roots<=1)
        roots = roots[mask]
        q_2 = roots[0]
        r_2 = alpha*coeff[2]**2/( Delta + coeff[2]**2/2*(1-q_2) + g(1) - g(q_W) )
        return q_2, r_2

    def eq(q_W): # reduce 4 equations to 1 equation in qw
        q_2, r_2 = q2r2(q_W)
        r_W = 1/gamma*r_2*q_W/(1+r_2*(1-q_W**2)) + alpha/gamma*g_prime(q_W)/( Delta + coeff[2]**2/2*(1-q_2) + g(1)-g(q_W) )
        return q_W - m(r_W)

    def potential(q_W):
        r_W = m_inv(q_W)
        q_2, r_2 = q2r2(q_W)
        return (-q_2*r_2/2 - alpha*np.log(Delta + coeff[2]**2*(1-q_2)/2 + g(1) - g(q_W)) + r_2/2 - gamma*q_W*r_W + 2*gamma*psi(r_W) - (1/2)*np.log(1 + r_2*(1-q_W**2) ) )/(2*alpha)

    # compute all quantities
    sols = bisect_roots(eq, 0, 1-1e-10)
    q_W = max(sols, key=potential)
    q_2, r_2 = q2r2(q_W)
    pot = potential(q_W)
    MI = (-np.log(Delta)/2 - pot)*alpha/gamma 
    mmse = coeff[2]**2/2*(1-q_2) + g(1) - g(q_W)
    return {'alpha': alpha,
            'mmse': mmse,
            'qw': q_W,
            'q2': q_2,
            'pot': pot,
            'MI': MI}

def mmse(activation, qw, q2):
    '''
    mmse in terms of qw, q2
    '''
    g, g_prime, coeff = gg(activation)
    return coeff[2]**2/2*(1-q2) + g(1) - g(qw)




    
    
