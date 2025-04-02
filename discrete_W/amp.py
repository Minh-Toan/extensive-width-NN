import numpy as np
from scipy.integrate import quad
from scipy.optimize import root


def solve_poly(z, sigma, kappa):
    alpha = 1 / kappa
    R_noise = sigma
    a3 = np.sqrt(alpha) * R_noise
    a2 = -(np.sqrt(alpha) * z + R_noise)
    a1 = (z + np.sqrt(alpha) - alpha**(-1 / 2))
    a0 = -1

    # Coefficients of the polynomial
    coefficients = [a3, a2, a1, a0]

    # Find the roots of the polynomial
    return np.roots(coefficients)


def edges_rho(sigma, kappa):
    alpha = 1/kappa
    R_noise = sigma

    a0 = -12 * R_noise + (4 * R_noise) / alpha + 12 * alpha * R_noise - 4 * alpha**2 * R_noise - 20 * R_noise**2 + R_noise**2 / alpha - 8 * alpha * R_noise**2 - 4 * R_noise**3
    a1 = -(10 * R_noise) / np.sqrt(alpha) + 2 * np.sqrt(alpha) * R_noise + 8 * alpha**(3/2) * R_noise - (2 * R_noise**2) / np.sqrt(alpha) + 8 * np.sqrt(alpha) * R_noise**2
    a2 = 1 - 2 * alpha + alpha**2 + 8 * R_noise - 2 * alpha * R_noise + R_noise**2
    a3 = -2 * np.sqrt(alpha) - 2 * alpha**(3/2) - 2 * np.sqrt(alpha) * R_noise
    a4 = alpha

    # Coefficients of the polynomial
    coefficients = [a4, a3, a2, a1, a0]

    roots_all = np.roots(coefficients)
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-6])

    return np.sort(real_roots)


def rho(x, sigma, kappa):        
    return np.max(np.imag(solve_poly(x-1e-8j, sigma, kappa))) / np.pi

def integral_rho(Delta, kappa):
        
    def rho(x):        
        return np.max(np.imag(solve_poly(x-1e-8j, Delta, kappa))) / np.pi

    
    edges_list = edges_rho(Delta, kappa)

    if len(edges_list) == 4:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1])[0] + quad(lambda x: rho(x)**3, edges_list[2], edges_list[3])[0]
    else:
        return quad(lambda x: rho(x)**3, edges_list[0], edges_list[1])[0]



def f_RIE(R, Delta, kappa):
    Delta = Delta + 1e-6
    def denoiser(x):        
        choose_root = np.argmax(np.imag(solve_poly(x-1e-8j, Delta, kappa))) 
        return np.real(solve_poly(x-1e-8j, Delta, kappa))[choose_root]
    
    eigval, eigvec = np.linalg.eig(R)
    eigval_denoised = np.array([e - 2*Delta*denoiser(e) for e in eigval])
    return eigvec @ np.diag(eigval_denoised) @ eigvec.T


def F_RIE(Delta, kappa):
    return Delta - 4*np.pi**2/3 * Delta**2 * integral_rho(Delta, kappa)


def AMP(X, y, gamma, noise, iterations, damping, tol):
    
    def gOut(y, w, V):
        return (y-w)/(noise + V)

    N, D, _ = X.shape
    r = int(D*gamma)
    alpha = N/D**2
        
    uX = X / np.sqrt(D) # X_mu has O(1) components and y_mu is O(1). We normalise X to have simpler equations later
        
    # hatS has O(1) SPECTRUM
    W = np.random.normal(0,1, (D, r))
    hatS = W @ W.T / np.sqrt(r) / np.sqrt(D) 
        
    hatC    = 10.
    omega   = np.ones(N)*10.
    V       = 10.

    error = np.inf
    for t in range(iterations):
        newV = hatC
        newOmega = np.einsum("nij,ij", uX, hatS) - gOut(y, omega, V) * newV
        
        V = newV * (1-damping) + V * damping
        omega = newOmega * (1-damping) + omega * damping
        
        # Factor 2
        A_normalised = np.sum(gOut(y, omega, V)**2) * alpha / N * 2
        R = hatS + 1 / (A_normalised * D)  * np.sum(gOut(y, omega, V)[:, None, None] * uX, axis=0)
        
        # Factor 2
        noise_A = 1 / A_normalised / 2
        newhatS = f_RIE(R, noise_A, r/D)
        hatC = F_RIE(noise_A, r/D)  * 2
        
        error = np.linalg.norm(hatS - newhatS)**2 / D
        hatS = newhatS
        
        if error < tol:
            break

    return np.sqrt(D)*hatS

