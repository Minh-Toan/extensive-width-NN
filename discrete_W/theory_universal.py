import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from activations import *

def Ez(f):
    integrand = lambda z: f(z)*np.exp(-z**2/2)/np.sqrt(2*np.pi)
    return quad(integrand, -10, 10)[0]

def coeffs(f):
    c0 = Ez(lambda z: f(z))
    c1 = Ez(lambda z: z*f(z))
    c2 = Ez(lambda z: (z**2-1)*f(z))/np.sqrt(2)
    nu = Ez(lambda z: f(z)**2)
    return c0, c1, c2, nu

def density(vlaw, coeff, ratio, x): 
    '''
    spectral density of coeff*W.T Dv W/N + Z
    W Gaussian size NxD, N/D = ratio
    Z GOE with supp [-2,2]
    vlaw: spectral law of Dv 
    examples of vlaw:
        'ones': v=(1,...,1)
        'rad' : Rademacher
        'uni' : Unif([-sqrt(3), sqrt(3)])
    '''
    c, a = coeff, ratio
    if vlaw == 'ones':
        p = [c, a + c*x, -a*c + a*x + c, a]
        den = np.max(np.imag(np.roots(p)))/np.pi
    if vlaw == 'rad':
        p = [-c**2, -c**2*x, a**2 + a*c**2 - c**2, a**2*x, a**2]
        den = np.max(np.imag(np.roots(p)))/np.pi
    if vlaw == 'uni':
        R = lambda s: -a/s + 1/(2*np.sqrt(3))*a**2/(c*s**2)*(np.log(a+np.sqrt(3)*c*s)-np.log(a-np.sqrt(3)*c*s)) + s # R-transform
        def eq(g): # equation satisfied by Stieltjes transform g(x)
            g = g[0] + 1j*g[1]
            eq = R(c, a, -g) - 1/g - x
            return [eq.real, eq.imag]
        eps = 1e-5
        if x>=0:
            den = np.abs(fsolve(eq, [-eps,eps])[1])/np.pi
        else:
            den = np.abs(fsolve(eq, [eps,eps])[1])/np.pi
    return den

def support(vlaw, coeff, ratio): 
    '''
    support of the density defined above
    '''
    c, a = coeff, ratio
    
    if vlaw == 'ones':
        p = [a**2*c**2,
             -2*a**3*c - 2*a**2*c**3 - 2*a*c**3,
             a**4 + 8*a**3*c**2 + a**2*c**4 - 2*a**2*c**2 - 2*a*c**4 + c**4,
             -2*a**4*c - 10*a**3*c**3 + 8*a**3*c + 2*a**2*c**3 + 8*a*c**3,
             a**4*c**2 - 4*a**4 + 4*a**3*c**4 - 20*a**3*c**2 - 12*a**2*c**4 - 8*a**2*c**2 + 12*a*c**4 - 4*c**4]
        sols = np.roots(p)
        criteria = np.abs(np.imag(sols))<1e-6
        edges = np.real(sols[criteria])
        xmin, xmax = np.min(edges), np.max(edges)
        
    if vlaw == 'rad':
        p = [a**4*c**4,
            -2*a**6*c**2 + 5*a**5*c**4 + a**4*c**6/4 - 8*a**4*c**4 - 5*a**3*c**6 - 2*a**2*c**6,
            a**8 + 3*a**7*c**2 + 3*a**6*c**4 + 12*a**6*c**2 + a**5*c**6 - 13*a**5*c**4 - 26*a**4*c**6 + 22*a**4*c**4 - a**3*c**8 + 13*a**3*c**6 + 3*a**2*c**8 + 12*a**2*c**6 - 3*a*c**8 + c**8,
            -4*a**8 - 16*a**7*c**2 - 24*a**6*c**4 - 16*a**6*c**2 - 16*a**5*c**6 - 16*a**5*c**4 - 4*a**4*c**8 + 16*a**4*c**6 - 24*a**4*c**4 + 16*a**3*c**8 + 16*a**3*c**6 - 24*a**2*c**8 - 16*a**2*c**6 + 16*a*c**8 - 4*c**8]
        sols = np.roots(p)
        criteria = np.abs(np.imag(sols))<1e-6
        edges = np.real(sols[criteria])
        xmin, xmax = -np.sqrt(np.max(edges)), np.sqrt(np.max(edges))

    if vlaw == 'uni':
        xmin, xmax = -4.5*c-2, 4.5*c + 2 # works for ratio = 0.5
        
    return xmin, xmax

def m(vlaw, ratio, l):
    '''
    overlap function of channel sqrt(l)*W.T Dv W/sqrt(N) + Z
    W Gaussian size NxD, N/D = ratio
    Z Gaussian symmetric
    '''
    if l==0:
        return 0
    c = np.sqrt(ratio*l)
    xmin, xmax = support(vlaw, c, ratio)
    integrand = lambda x: density(vlaw, c, ratio, x)**3
    integral = (4*np.pi**2/3)*quad(integrand, xmin, xmax)[0]
    return 1-(1-integral)/l


def quantities(vlaw, activation, alpha, gamma, Delta):
    '''
    compute q2, free energy, mutual information, mmse, etc
    '''
    phi = lookup(activation)
    c0, c1, c2, nu = coeffs(phi)
    if np.abs(c2) < 1e-10: # c2=0
        q2 = 0
        pot = - np.log(Delta + nu - c0**2 - c1**2)/2
        MI = (-np.log(Delta)/2 - pot)*alpha/gamma
        mmse = nu - c0**2 - c1**2
    else:
        Delta_2 = (Delta + nu - c0**2 - c1**2 - c2**2)/c2**2
        def eq(r):
            return m(vlaw, gamma, r[0]) - 1 - Delta_2 + 2*alpha/r[0]
        init = 2*alpha/(0.5 + Delta_2)
        r2 = fsolve(eq, init)[0]
        q2 = 1 + Delta_2 - 2*alpha/r2
        pot  = -r2*q2/(4*alpha) - np.log(Delta + nu - c0**2 - c1**2 - c2**2*q2 )/2 + quad(lambda r: m(vlaw, gamma, r), 0, r2)[0]/(4*alpha)
        MI = (-np.log(Delta)/2 - pot)*alpha/gamma
        mmse = nu - c0**2 - c1**2 - c2**2*q2
    return {'alpha': alpha,
            'mmse': mmse,
            'q2': q2,
            'pot': pot,
            'MI': MI}

def mmse(phi, q2):
    '''
    mmse from empirical q2
    '''
    c0, c1, c2, nu = coeffs(phi)
    return nu - c0**2 - c1**2 - c2**2*q2
    













    