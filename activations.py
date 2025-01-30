import numpy as np
from scipy.integrate import quad

def hermite_coefficients(func, n, a, b):
    """
    Computes the first n Hermite coefficients of a function.

    Args:
        func: The function whose coefficients are to be computed.
        n: The number of coefficients to compute.
        a: The lower limit of the integration interval.
        b: The upper limit of the integration interval.

    Returns:
        A NumPy array containing the first n Hermite coefficients.
    """

    # Define the Hermite polynomials
    def hermite(x, n):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return x * hermite(x, n - 1) - (n - 1) * hermite(x, n - 2)

    # Define the weight function
    def weight(x):
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)

    # Compute the coefficients
    coefficients = []
    for i in range(n+1):
        # Use scipy.integrate.quad to compute the integral numerically.
        # Note: The integrand needs to be defined as a lambda function.
        integral, error = quad(lambda x: func(x) * hermite(x, i) * weight(x), a, b)
        coefficients.append(integral)

    return np.array(coefficients)

def lookup(activation):
    if (activation == 'ReLU'):
        return lambda x: np.maximum(x,0)
        
    if (activation == 'Tanh'):
        return lambda x: np.tanh(x)

    if (activation == 'LeakyReLU'):
        return lambda x: np.heaviside(x,1)*x + 0.3*np.heaviside(-x,-1)*x

    if (activation == 'ELU'):
        return lambda x: np.heaviside(x,1)*x + np.heaviside(-x,1)*(np.exp(x) - 1)

    if (activation == 'Sigmoid'):
        return lambda x: 1/(1+np.exp(-x))

    if (activation == 'Quadratic'):
        return lambda x: np.power(x,2)/np.sqrt(2)

    if (activation == 'Quadratic + Cubic'):
        return lambda x: (x**2 - 1)/np.sqrt(2) + (x**3 - 3*x)/6

    if (activation == 'Linear + Cubic'):
        return lambda x: (x**3 - 3*x)/np.sqrt(6)
    

def gg(activation):
    num_coeff = 15
    sigma = lookup(activation)
    coefficients = hermite_coefficients(sigma, num_coeff, -10, 10)
    
    # if (activation == 'ReLU'):
    #     g = lambda x: -1/(2*np.pi) - x**2/(4*np.pi) + np.sqrt(1-x**2)/(2*np.pi) + x*np.arctan(x/np.sqrt(1-x**2))/(2*np.pi)
    #     g_prime = lambda x: (np.arctan(x/np.sqrt(1-x**2))-x)/(2*np.pi)
    
    # Create the approximation function using the computed coefficients
    def g(x):
        g = 0
        for i in range(3,len(coefficients)):
            g += coefficients[i]**2 * x**i / np.math.factorial(i)
        return g

    def g_prime(x):
        g = 0
        for i in range(3,len(coefficients)):
            g += coefficients[i]**2 * x**(i-1) / np.math.factorial(i-1)
        return g
    return g, g_prime, coefficients