'''
faster computation;
however, computations are limited for gamma=0.5
'''
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, brentq
import activations as act

from scipy.interpolate import CubicSpline

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_ones = np.loadtxt(script_dir+'/stored_func/gamma_0.5/ones.csv', delimiter=',')
data_rad  = np.loadtxt(script_dir+'/stored_func/gamma_0.5/rad.csv', delimiter=',')
data_unif = np.loadtxt(script_dir+'/stored_func/gamma_0.5/unif.csv', delimiter=',')
data_gauss = np.loadtxt(script_dir+'/stored_func/gamma_0.5/gauss.csv', delimiter=',')

mS_ones  = CubicSpline(data_ones[0] , data_ones[1] )
mS_rad  = CubicSpline(data_rad[0] , data_rad[1] )
mS_unif = CubicSpline(data_unif[0], data_unif[1])
mS_gauss = CubicSpline(data_gauss[0], data_gauss[1])

psiS_ones  = CubicSpline(data_ones[0] , data_ones[2] )
psiS_rad  = CubicSpline(data_rad[0] , data_rad[2] )
psiS_unif = CubicSpline(data_unif[0], data_unif[2])
psiS_gauss = CubicSpline(data_gauss[0], data_gauss[2])

imS_ones = CubicSpline(data_ones[1], data_ones[0])
imS_rad  = CubicSpline(data_rad[1] , data_rad[0] )
imS_unif = CubicSpline(data_unif[1], data_unif[0])
imS_gauss = CubicSpline(data_gauss[1], data_gauss[0])

mS_map = {'ones': mS_ones,
          'rad': mS_rad, 'radd': mS_rad,
          'unif': mS_unif, 'unid': mS_unif,
          'gauss': mS_gauss, 'gaussd': mS_gauss
         }

imS_map = {'ones': imS_ones,
           'rad': imS_rad, 'radd': imS_rad,
           'unif': imS_unif, 'unid': imS_unif,
           'gauss': imS_gauss, 'gaussd': imS_gauss
          }

psiS_map = {'ones': psiS_ones,
            'rad': psiS_rad, 'radd': psiS_rad,
            'unif': psiS_unif, 'unid': psiS_unif,
            'gauss': psiS_gauss, 'gaussd': psiS_gauss
           }

def mS(vlaw, l):
    return mS_map[vlaw](l)

def psiS(vlaw, l):
    return psiS_map[vlaw](l)

def imS(vlaw, q):
    return imS_map[vlaw](q)

def d_imS(vlaw, q):
    return imS_map[vlaw].derivative()(q)
    
    
def all_roots(f, a, b):
    '''
    Find all roots in [a,b] of a function f
    '''
    x_vals = np.linspace(a, b, 100)
    f_vals = [f(x_val) for x_val in x_vals]
    # Find intervals where the function changes sign
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    roots = []
    for i in sign_changes:
        try:
            root = brentq(f, x_vals[i], x_vals[i + 1])
            roots.append(root)
        except ValueError:
            pass  
    return roots

def m(l, prior):
    '''
    overlap E[XE[X|Y]] of the Gaussian channel Y = sqrt(l)*X + Z
    '''
    if prior=='gauss':
        return l/(l+1)
    if prior=='rad':
        integrand = lambda z: np.tanh(np.sqrt(l)*z + l)*np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return quad(integrand, -10, 10)[0]

def m_inv(q, prior):
    '''
    inverse of function m
    '''
    if prior=='gauss':
        return q/(1-q)
    if prior=='rad':
        u = 1
        mm = lambda l: m(l, prior)
        if mm(u)>=q:
             r = bisect(lambda x: mm(x)-q, 0, 1)
        else:
            while mm(u)<q:
                u *= 2
            r = bisect(lambda x: mm(x)-q, u/2, u)
        return r

def psi(l, prior):
    '''
    free energy function of the Gaussian channel sqrt(l)*X + Z
    '''
    if prior=='gauss':
        return l/2 - np.log(1+l)/2
    if prior=='rad':
        integrand = lambda z: np.log( np.cosh( np.sqrt(l)*z + l ) )*np.exp( -z**2/2 )/np.sqrt(2*np.pi)
        return -l/2 + quad(integrand, -10, 10)[0]


def sp_quantities(sig, alpha, gamma, Delta, vlaw, prior):
    g, g_prime, mu = act.gg(sig)
            
    def q2r2(qw):
        l = qw**2/(1-qw**2)
        eq = lambda q: q - m(l + alpha*mu[2]**2/(Delta + mu[2]**2/2*(1-q) + g(1) - g(qw)), 'gauss')
        q2 = brentq(eq, 0, 0.996)
        r2 = alpha*mu[2]**2/(Delta + mu[2]**2/2*(1-q2) + g(1) - g(qw))
        return q2, r2
        
    def eq(qw): # reduce 4 equations to 1 equation in qw
        q2, r2 = q2r2(qw)
        rw = alpha/gamma*g_prime(qw)/(Delta + mu[2]**2/2*(1-q2) + g(1)-g(qw)) + (q2-qw**2)*qw/(gamma*(1-qw**2)**2)
        return qw - m(rw, prior)

    def potential(qw):
        l = qw**2/(1-qw**2)
        rw = m_inv(qw, prior)
        q2, r2 = q2r2(qw)
        return -alpha*np.log(Delta+mu[2]**2/2*(1-q2)+g(1)-g(qw)) + 2*gamma*psi(rw, prior) - gamma*qw*rw + psi(l+r2, 'gauss') - psi(l, 'gauss') - q2*r2/2
        
    sols = all_roots(eq, 0, 0.996)
    qw = max(sols, key=potential)
    q2, r2 = q2r2(qw)
    pot = potential(qw)/(2*alpha)
    MI = (-np.log(Delta)/2 - pot)*alpha/gamma 
    mmse = mu[2]**2/2*(1-q2) + g(1)-g(qw)
    
    return {'alpha': alpha,
            'qw':qw,
            'q2':q2,
            'mmse':mmse,
            'MI':MI
            }

def uni_quantities(vlaw, sig, alpha, gamma, Delta):
    g, g_prime, mu = act.gg(sig)
    eq = lambda q: q - mS(vlaw, alpha*mu[2]**2/(Delta + mu[2]**2/2*(1-q) + g(1)))
    q2 = brentq(eq, 0, 0.996)
    r2 = alpha*mu[2]**2/(Delta + mu[2]**2/2*(1-q2) + g(1))
    mmse = mu[2]**2/2*(1-q2) + g(1)
    pot= -r2*q2/(4*alpha) - np.log(Delta + mu[2]**2/2*(1-q2) + g(1))/2 + psiS(vlaw, r2)/(4*alpha)
    MI = (-np.log(Delta)/2 - pot)*alpha/gamma
    
    return {'alpha': alpha,
            'q2':q2,
            'mmse':mmse,
            'MI':MI
           }

    




    













    