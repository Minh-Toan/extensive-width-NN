import numpy as np
import math
from scipy.integrate import quad
from scipy.special import erf

def hermite_coefficients(func, n, a, b):
    def hermite(x, n):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return x * hermite(x, n - 1) - (n - 1) * hermite(x, n - 2)

    def weight(x):
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)

    coefficients = []
    for i in range(n+1):
        integral, error = quad(lambda x: func(x) * hermite(x, i) * weight(x), a, b)
        coefficients.append(integral)
        
    return np.array(coefficients)

def lookup(activation):
    if (activation == 'relu'):
        return lambda x: np.maximum(x,0)
        
    if (activation == 'tanh'):
        return lambda x: np.tanh(x)

    if (activation == 'erf2'):
        return lambda x: erf(2*x)

    if (activation == 'elu'):
        return lambda x: np.heaviside(x,1)*x + np.heaviside(-x,1)*(np.exp(x) - 1)

    if (activation == 'sigmoid'):
        return lambda x: 1/(1+np.exp(-x))

    if (activation == 'h2'):
        return lambda x: np.power(x,2)/np.sqrt(2)

    if (activation == 'h23'):
        return lambda x: (x**2 - 1)/np.sqrt(2) + (x**3 - 3*x)/6

    if (activation == 'h3'):
        return lambda x: (x**3 - 3*x)/np.sqrt(6)
    

def gg(sigma):
    num_coeff = 15
    coefficients = hermite_coefficients(sigma, num_coeff, -10, 10)
    # special handling for relu
    x = np.array([-2, -1, 0, 1, 2])
    if np.allclose([sigma(xi) for xi in x], np.maximum(x, 0)):
        def g(x):
            if x==1:
                return 1/4*(1 - 3/(np.pi))
            return -1/(2*np.pi) - x**2/(4*np.pi) + np.sqrt(1-x**2)/(2*np.pi) + x*np.arctan(x/np.sqrt(1-x**2))/(2*np.pi)
        def g_prime(x):
            if x==1:
                return 1/4 - 1/(2*np.pi)
            return (np.arctan(x/np.sqrt(1-x**2))-x)/(2*np.pi)
        return g, g_prime, coefficients
    
    def g(x):
        g = 0
        for i in range(3,len(coefficients)):
            g += coefficients[i]**2 * x**i / math.factorial(i)
        return g

    def g_prime(x):
        g = 0
        for i in range(3,len(coefficients)):
            g += coefficients[i]**2 * x**(i-1) / math.factorial(i-1)
        return g
    return g, g_prime, coefficients

