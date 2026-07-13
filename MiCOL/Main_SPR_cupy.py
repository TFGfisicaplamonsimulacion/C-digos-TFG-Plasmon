#%% Importar librerias
import cupy as cp
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\Instrumentación_Laboratorio\\Simulacion')
from Indices import Index_cupy as ri

#%% Variables generales

deg = 180 /cp.pi # [º/rad]

#%% Angulos y criticos

def varphi2theta(var_phi: float , ldo:float , prism: str, alpha: float) -> float:
    '''
    Parameters
    ----------
    var_phi : float
        angulo de entrada del prisma [grados].
    ldo : float
        Lonxitude de onda [nm].
    prism : str
        Material del prisma.
    alpha : float
        angulo alpha del prisma [grados].

    Returns
    -------
    theta_int : float
        angulo interno [grados].

    '''
    
    nair = ri.refindex(ldo, material = 'air').real
    cprism = ri.refindex(ldo, material = prism).real
    
    aux = (nair / cprism) / cp.sin (var_phi / deg)
    theta_int = alpha + cp.arcsin(aux)/deg
    
    return theta_int

def theta2varphi(theta_int: float, ldo: float, prism: str, alpha:float) -> float:
    '''
    Parameters
    ----------
    theta_int : float
        angulo interno [grados].
    ldo : float
        Lonxitude de onda [nm].
    prism : str
        Material del prisma.
    alpha : float
        angulo alpha del prisma [grados].

    Returns
    -------
    var_phi : float
        angulo de entrada del prisma [grados].

    '''
    nair = ri.refindex(ldo, material = 'air').real
    cprism = ri.refindex(ldo, material = prism).real
    
    aux = cp.sin ( (theta_int - alpha) / deg )
    var_phi = cp.arcsin( cprism / nair * aux ) *deg
    
    return var_phi
    
def theta_cr(ldo: float, mat_1: str, mat_2: str) -> float:
    '''
    Parameters
    ----------
    ldo : float
        Lonxitude de onda [nm].
    mat_1 : str
        Material 1.
    mat_2 : str
        Material 2.

    Returns
    -------
    th_cr: float
        Angulo critico [grados]

    '''
    n1 = ri.refindex(ldo, material = mat_1)
    n2 = ri.refindex(ldo, material = mat_2)
    
    th_cr = cp.arcsin( n2.real / n1.real ) * deg
    if cp.isnan(th_cr).any():
        print('Critical angle not found, check materials\n')
    
    return th_cr

def varphi_cr(ldo: float, mat_1: str, mat_2: str, alpha:float) -> float:
    '''
    Parameters
    ----------
    ldo : float
        Lonxitude de onda [nm].
    mat_1 : str
        Material 1.
    mat_2 : str
        Material 2.
    alpha : float
        angulo alpha del prisma [grados].

    Returns
    -------
    vphi_cr: float
        angulo critico de entrada del prisma [grados].

    '''
    
    th_cr = theta_cr(ldo, mat_1, mat_2)
    vphi_cr = theta2varphi(th_cr, ldo, mat_1, alpha)
    
    return vphi_cr

#%% SPR functions

def Multicapa_Matrix(ldo: float, mats: list, d: float, theta_i:float, Polarization = 'TM'):
    
    cp._default_memory_pool.free_all_blocks()
    
    ldo = cp.array(ldo)
    d = cp.array(d)
    theta_i = cp.array(theta_i).flatten()/deg
    
    index_eval = cp.array([ri.refindex(ldo, mat) for mat in mats])
    index_0 = cp.array([ri.refindex(ldo, mats[0]) for mat in mats])
    
    N = theta_i.shape[0]
    M, K = index_eval.shape
    
    
    
    # Expand their dimenssions
    d_3D = cp.expand_dims(d, axis=(0, 2))   # Shape (1, M, 1)
    theta_3D = cp.expand_dims(theta_i, axis=(1, 2))   # Shape (N, 1, 1)
    index_3D = cp.expand_dims(index_eval, axis=0)     # Shape (1, M, K)
    ldo_3D = cp.expand_dims(ldo, axis=(0,1))     # Shape (1, 1, K)
    
    index_0 = cp.expand_dims(index_0, axis=0)     # Shape (1, M, K)
    
    
    
    # Now broadcast x and Y to form the 3D matrices
    d_3D = cp.tile(d_3D, (N, 1, K))         # Shape (N, M, K)
    theta_3D = cp.tile(theta_3D, (1, M, K))         # Shape (N, M, K)
    index_3D = cp.tile(index_3D, (N, 1, 1))      # Shape (N, M, K)
    ldo_3D = cp.tile(ldo_3D, (N, M, 1))      # Shape (N, M, K)
    
    index_0 = cp.tile(index_0, (N, 1, 1))      # Shape (N, M, K)
    
    sin_theta_r = cp.sin(theta_3D) * index_0/index_3D
    cos_theta_r = cp.sqrt((1.+0j) - sin_theta_r**2)
    
    # #segundo o tipo de polarización cambia o cálculo de índice efectivo
    if Polarization.lower() == 'tm':
        u = index_3D / cos_theta_r
    elif Polarization.lower() == 'te': 
        u = index_3D * cos_theta_r
    
    # Número de ondas no baleiro
    k0 = 2. * cp.pi / ldo_3D
    # Fase adquirida ao atravesar cada unha das capas
    g = k0 * index_3D * d_3D * cos_theta_r

    
    # Create a 2x2 identity matrix
    identity_matrix = cp.eye(2, dtype=complex)
    
    # Repeat the identity matrix along the new dimensions (N for angles, K for wavelengths)
    Ms = cp.repeat(identity_matrix[None, :, :, None], N, axis=0)  # Repeat along the first axis (N)
    Ms = cp.repeat(Ms, K, axis=-1)  # Repeat along the last axis (K)
    
    # Compute the cumulative transfer matrix across layers
    for i in range(1, M - 1): 
        M_layer = Mnew(g[:, i, :], u[:, i, :])  # Shape (N, 2, 2, K)
        # Perform matrix multiplication for each angle and wavelength
        Ms = cp.einsum('nijw,njkw->nikw', Ms, M_layer)  # Shape remains (N, 2, 2, K)
    
    
    
    A_matrices = A(Ms, u[:, 0, :], u[:, -1, :])
    A_matrices = cp.transpose(A_matrices, (1, 3, 2, 0))   # Reorder to (N, 2, 2, K)
    B_vector = B(N,K)

    
    X = cp.linalg.solve(A_matrices, B_vector)
    
    
    t = X[:, :, 0]
    r = X[:, :, 1]
    
    R = cp.abs(r) ** 2
    
    if Polarization.lower() == 'tm':
        T = (index_3D[:, -1, :] * cos_theta_r[:, 0, :]) / (index_3D[:, 0, :] * cos_theta_r[:, -1, :]) * cp.abs(t) ** 2
    else:
        T = (index_3D[:, -1, :] * cos_theta_r[:, -1, :]) / (index_3D[:, 0, :] * cos_theta_r[:, 0, :]) * cp.abs(t) ** 2

    
    return r, t, R, T
#%% Funciones auxiliares

def Mnew(g,u):
    M00 = cp.cos(g)
    M01 = -1j * cp.sin(g) / u 
    M10 = -1j * cp.sin(g) * u   
    M11 = cp.cos(g)
   # Stack these into a 4D array and transpose to get the shape (N, 2, 2, K)
    M_new = cp.array([[M00, M10], [M01, M11]])  # Shape (2, 2, N, K)
    M_new = cp.transpose(M_new, (2, 1, 0, 3))   # Reorder to (N, 2, 2, K)    
    return M_new

def A(M, u0, uf):
    A00 = M[:, 0, 0] + M[:, 0, 1] * uf
    A01 = cp.full_like(A00, -1.0) 
    A10 = (M[:, 1, 0] + M[:, 1, 1] * uf) / u0
    A11 = cp.full_like(A00, 1.0)
  
    A = cp.stack([cp.stack([A00, A01], axis=-1), cp.stack([A10, A11], axis=-1)], axis=1)
    return cp.moveaxis(A, -1, 0)

def B(N,K):
    return cp.ones([N,K,2])

