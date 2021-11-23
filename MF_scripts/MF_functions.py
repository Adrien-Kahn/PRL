import time

import numpy as np
import numpy.linalg as la
from scipy.optimize import brentq, minimize_scalar


#--------PARAMETERS--------#

#Lattice definition
t = 1.0
tp1 = 1.0
tp2 = 0
tperp = 0.
Lx, Ly, Lz = 128, 128, 1

# dispersion and k-mesh
epsk = lambda kx, ky, kz: - 2 * t * (np.cos(kx) + np.cos(ky)) \
    - 2 * tp1 * np.cos(kx + ky)         \
    - 2 * tp2 * np.cos(kx - ky)         \
    - 2 * tperp * np.cos(kz)

kxr = np.linspace(-np.pi, np.pi, Lx, endpoint=False)
kyr = np.linspace(-np.pi, np.pi, Ly, endpoint=False)
kzr = np.linspace(-np.pi, np.pi, Lz, endpoint=False)
kx, ky, kz = np.meshgrid(kxr,kyr,kzr)



U = 2



#--------- Mean field equations ---------#
# fermi function
def fermi(eps, beta):
    return 1 / (1 + np.exp(beta * eps))

# derivative of fermi function
def dfermi(eps, beta):
    return -beta / (2 * np.cosh(beta*eps) + 2)

def n_and_slope(mup, nn, mm, Q):
    
    Qx, Qy, Qz = Q
    
    """Returns the density and the slope of the magnetization"""
    Ep = epsk(kx+Qx/2, ky+Qy/2, kz+Qz/2)
    Em = epsk(kx-Qx/2, ky-Qy/2, kz-Qz/2)
    Ek = 0.5 * (Ep + Em)
    etak = 0.5 * (Ep - Em)
    delta = U * mm / 2
    
    E_plus  = Ek + U * nn / 2 - mup + np.sqrt(etak**2 + delta**2)
    E_minus = Ek + U * nn / 2 - mup - np.sqrt(etak**2 + delta**2)
    
    # treatment to avoid divisions by zero
    M = np.zeros_like(kx)
    delta_E = E_plus - E_minus
    mask = np.abs(E_plus - E_minus) > 1e-10
    delta_E[~mask] = np.nan
    M[mask] = ((fermi(E_minus, beta) - fermi(E_plus, beta)) / delta_E)[mask]
    M[~mask] = (-dfermi(E_plus, beta))[~mask]
    
    n = np.sum(fermi(E_plus, beta) + fermi(E_minus, beta))
    slope = U * np.sum(M)
    n /= Lx*Ly*Lz
    slope /= Lx*Ly*Lz
    return [n, slope]

def solve_for_n(mu, m, Q):
    """Returns n and the slope"""
    n = brentq(lambda n1: n_and_slope(mu, n1, m, Q)[0] - n1, 0, 2)
    return n_and_slope(mu, n, m, Q)

def solve_for_mu(density, m, Q):
    """Returns mu, n and the slope"""
    mu = brentq(lambda mu1: solve_for_n(mu1, m, Q)[0] - density, -5*U, 5*U)
    return [mu] + solve_for_n(mu, m, Q)

def solve_mean_field(density, Q):
    """Returns the density, the magnetization and the chemical potential"""
    mu, n, slope = solve_for_mu(density, 0.0, Q)
    if slope < 1: return n, 0.0, mu
    
    m = brentq(lambda m1: solve_for_mu(density, m1, Q)[2] - 1, 0., 1.)
    mu, n, slope = solve_for_mu(density, m, Q)
    return n, slope*m, mu

def compute_free_energy(g_or_h, n, m, mu, Q):
    
    Qx, Qy, Qz = Q
    
    Ep = epsk(kx+Qx/2, ky+Qy/2, kz+Qz/2)
    Em = epsk(kx-Qx/2, ky-Qy/2, kz-Qz/2)
    Ek = 0.5 * (Ep + Em)
    etak = 0.5 * (Ep - Em)
    delta = U * m / 2
    
    E_plus  = Ek + U * n / 2 - mu + np.sqrt(etak**2 + delta**2)
    E_minus = Ek + U * n / 2 - mu - np.sqrt(etak**2 + delta**2)
    
    # log Z term
    logZ =  - np.sum(np.log(np.ones(kx.shape) + np.exp(-beta*E_plus)))/beta   \
        - np.sum(np.log(np.ones(kx.shape) + np.exp(-beta*E_minus)))/beta
    
    # free energy
    if g_or_h == "Helmholtz":
        F = logZ / (Lx*Ly*Lz) + mu * n - 0.25 * U * (n-m) * (n+m)
    elif g_or_h == "Gibbs":
        F = logZ / (Lx*Ly*Lz) - 0.25 * U * (n-m) * (n+m)
    else:
        return np.nan
    
    return F




#------- Compute a point of the phase diagram -------#

#Functions that will be optimized in 2D
#Q along Ox
def opti_q0(q,n):
    Q = q, 0., 0.
    n_res, m, mu = solve_mean_field(n, Q)
    return (m != 0.)*compute_free_energy('Helmholtz', n_res, m, mu, Q)

#Q along (π,y)
def opti_piq(q,n):
    Q = np.pi, q, 0.
    
    n_res, m, mu = solve_mean_field(n, Q)
    
    return (m != 0.)*compute_free_energy('Helmholtz', n_res, m, mu, Q)

#Q along first bissectrix
def opti_qq(q,n):
    Q = q, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)
    return (m != 0.)*compute_free_energy('Helmholtz', n_res, m, mu, Q)

def compute_point(U,n,T,res_dict):
    
    global beta
    beta = 1/T
    
    #Compute paramagnetic case first
    Q = 0.,0.,0.
    mu, n_res, slope = solve_for_mu(n, 0.0, Q) # impose m=0
    
    res_dict[n,T,"energy","para"] = compute_free_energy('Helmholtz', n_res, 0.0, mu, Q)
    res_dict[n,T,"mag","para"] = 0
    res_dict[n,T,"mu","para"] = mu
    res_dict[n,T,"Q","para"] = 0
    
    #Then investigate Q vectors
    #(q,0)
    q = minimize_scalar(lambda q:opti_q0(q,n), bounds=(0, np.pi), method='bounded').x
    Q = q, 0., 0.
    n_res, m, mu = solve_mean_field(n, Q)
    
    res_dict[n,T,"energy","q0"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[n,T,"mag","q0"] = m
    res_dict[n,T,"mu","q0"] = mu
    res_dict[n,T,"Q","q0"] = q
    
    #(π,q)
    q = minimize_scalar(lambda q:opti_piq(q,n), bounds=(0, np.pi), method='bounded').x
    Q = np.pi, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)
    
    res_dict[n,T,"energy","piq"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[n,T,"mag","piq"] = m
    res_dict[n,T,"mu","piq"] = mu
    res_dict[n,T,"Q","piq"] = q
    
    #(q,q)
    q = minimize_scalar(lambda q:opti_qq(q,n), bounds=(0, np.pi), method='bounded').x
    Q = q, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)
    res_dict[n,T,"energy","qq"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[n,T,"mag","qq"] = m
    res_dict[n,T,"mu","qq"] = mu
    res_dict[n,T,"Q","qq"] = q

def compute_point_UT(U,n,T,res_dict):
    
    global beta
    beta = 1/T
    
    #Compute paramagnetic case first
    Q = 0.,0.,0.
    mu, n_res, slope = solve_for_mu(n, 0.0, Q) # impose m=0
    
    res_dict[U,T,"energy","para"] = compute_free_energy('Helmholtz', n_res, 0.0, mu, Q)
    res_dict[U,T,"mag","para"] = 0
    res_dict[U,T,"mu","para"] = mu
    res_dict[U,T,"Q","para"] = 0

    #Then investigate Q vectors
    #(q,0)
    q = minimize_scalar(lambda q:opti_q0(q,n), bounds=(0, np.pi), method='bounded').x
    Q = q, 0., 0.
    n_res, m, mu = solve_mean_field(n, Q)
    
    res_dict[U,T,"energy","q0"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[U,T,"mag","q0"] = m
    res_dict[U,T,"mu","q0"] = mu
    res_dict[U,T,"Q","q0"] = q

    #(π,q)
    q = minimize_scalar(lambda q:opti_piq(q,n), bounds=(0, np.pi), method='bounded').x
    Q = np.pi, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)
    
    res_dict[U,T,"energy","piq"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[U,T,"mag","piq"] = m
    res_dict[U,T,"mu","piq"] = mu
    res_dict[U,T,"Q","piq"] = q

    #(q,q)
    q = minimize_scalar(lambda q:opti_qq(q,n), bounds=(0, np.pi), method='bounded').x
    Q = q, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)
    res_dict[U,T,"energy","qq"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[U,T,"mag","qq"] = m
    res_dict[U,T,"mu","qq"] = mu
    res_dict[U,T,"Q","qq"] = q


def compute_point_full_grid(U,n,T,qx,qy,res_dict):
   
    global beta
    beta = 1/T
 
    Q = qx,qy,0.
    n_res, m, mu = solve_mean_field(n,Q)
    res_dict[qx,qy,"energy"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[qx,qy,"mag"] = m
    res_dict[qx,qy,"mu"] = mu

def compute_point_120_degrees(U,n,T,res_dict):

    global beta
    beta = 1/T

     #Compute paramagnetic case first
    Q = 0.,0.,0.
    mu, n_res, slope = solve_for_mu(n, 0.0, Q) # impose m=0

    res_dict[U,T,"energy","para"] = compute_free_energy('Helmholtz', n_res, 0.0, mu, Q)
    res_dict[U,T,"mag","para"] = 0
    res_dict[U,T,"mu","para"] = mu
    res_dict[U,T,"Q","para"] = 0

    #Then investigate Q vectors
    #(2π/3,2π/3)
    q = 2*np.pi/3
    Q = q, q, 0.
    n_res, m, mu = solve_mean_field(n, Q)

    res_dict[U,T,"energy","qq"] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
    res_dict[U,T,"mag","qq"] = m
    res_dict[U,T,"mu","qq"] = mu
    res_dict[U,T,"Q","qq"] = q



def q_type_to_Q(q,q_type):
    x = 0 if q_type[0] == '0' else (np.pi if q_type[0] == 'pi' else q)
    y = 0 if q_type[1] == '0' else (np.pi if q_type[1] == 'pi' else q)
    z = 0 if q_type[2] == '0' else (np.pi if q_type[2] == 'pi' else q)
    return x,y,z

def opti_3D(q,q_type,n):
    Q = q_type_to_Q(q,q_type)
    n_res, m, mu = solve_mean_field(n, Q)
    return (m != 0.)*compute_free_energy('Helmholtz', n_res, m, mu, Q)

def compute_point_3D(U,n,T,res_dict):
    
    global beta
    beta = 1/T
    
    #Compute paramagnetic cases
    Q = 0.,0.,0.
    mu, n_res, slope = solve_for_mu(n, 0.0, Q) # impose m=0
    
    res_dict[n,T,"energy","para"] = compute_free_energy('Helmholtz', n_res, 0.0, mu, Q)
    res_dict[n,T,"mag","para"] = 0
    res_dict[n,T,"mu","para"] = mu
    res_dict[n,T,"Q","para"] = 0
    
    #Then investigate Q vectors
    q_types = [('0','0','q'), ('0','q','q'), ('q','q','q'), ('pi','q','q'), ('pi','pi','q'), ('0','pi','q')]
    
    for q_type in q_types:
        q = minimize_scalar(lambda q:opti_3D(q,q_type,n), bounds=(0, np.pi), method='bounded').x
        Q = q_type_to_Q(q,q_type)
        n_res, m, mu = solve_mean_field(n, Q)
        
        phase_name = q_type[0] + q_type[1] + q_type[2]
        
        res_dict[n,T,"energy",phase_name] = compute_free_energy('Helmholtz', n_res, m, mu, Q)
        res_dict[n,T,"mag",phase_name] = m
        res_dict[n,T,"mu",phase_name] = mu
        res_dict[n,T,"Q",phase_name] = q
