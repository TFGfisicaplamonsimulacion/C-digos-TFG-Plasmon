#%% Importar librerias
import numpy as np
# import cupy as np
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela (1)\\Instrumentación_Laboratorio\\Simulacion')
sys.path.insert(2, 'C:\\Users\Admin\Desktop\Clase (falta pasar el pc antiguo aquí)\Cuarto\TFG\Programas')
from Indices import Index_no_va as ri

#%% Variables generales

deg = 180/np.pi # [º/rad]

#%% Angulos y criticos

def varphi2theta(var_phi: float, ldo: float, prism: str, alpha: float) -> float:
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
    nprism = ri.refindex(ldo, material = prism).real
    
    aux = (nair / nprism) * np.sin (var_phi / deg)
    theta_int = alpha + np.arcsin(aux) * deg
    
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
    nprism = ri.refindex(ldo, material = prism).real
    
    aux = np.sin ( (theta_int - alpha) / deg )
    var_phi = np.arcsin( nprism / nair * aux ) * deg
    
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
    
    th_cr = np.arcsin( n2.real / n1.real ) * deg
    if np.isnan(th_cr).any():
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



#%% SPR functions, multilayer calculation

def Multicapa_Matrix(ldo: float, mats: list , tempera: float, d: float, theta_i: float, Polarization = 'TM', Campos = False):

    N, M, K, index_3D, sin_theta_r, cos_theta_r, u, g = Common_Matrix(ldo = ldo, mats = mats,d = d ,tempera = tempera, theta_i = theta_i, Polarization = Polarization)
    
    Ms = Create_Multilayer_Matrix(N = N, M = M, K = K, g = g, u = u)
    
    #r,t Fresnel coeficients of the whole system
    A_matrices = A(Ms, u[:, 0, :], u[:, -1, :])
    A_matrices = np.transpose(A_matrices, (1, 3, 2, 0))   # Reorder to (N, 2, 2, K)
    B_vector = B(N,K)
    # AX=B => X = A^{-1} B    
    X = np.linalg.solve(A_matrices, B_vector)
    t = X[:, :, 0]
    r = X[:, :, 1]

    #reflectance and transmittance of the whole system
    R = np.abs(r) ** 2      
    if Polarization.lower() == 'tm':
        T = (index_3D[:, -1, :] * cos_theta_r[:, 0, :]) / (index_3D[:, 0, :] * cos_theta_r[:, -1, :]) * np.abs(t) ** 2
    else:
        T = (index_3D[:, -1, :] * cos_theta_r[:, -1, :]) / (index_3D[:, 0, :] * cos_theta_r[:, 0, :]) * np.abs(t) ** 2

    if Campos == True:        
        #creamos a matriz na que almacear os campos, partindo do resultado na primeira capa
        #partimos da matriz de r e engadimos unha dimension máis en cada elemento (4 compoñentes do campo no medio 0)
        #Creamos dúas matrices (vector) auxiliares e repetímolas nas dúas dimensións
        aux1 = np.array([[0.],[1.],[0.],[1.]])
        aux2 = np.ones((4,1)) - aux1  
        Fields_x = increase_dims(r) * expandNK(aux1,N,K) + expandNK(aux2,N,K)
        #a matriz campos (tanxenciais) ten para cada angulo e cada ldo un vector de catro elementos:
        #os campos no primeiro medio: E0l_p, E0l_m, E0_p, E0_m
        for i in range(M-1):       
             # Aplicamos a función ao longo dos eixos (B, C)
             new_fields = Campos_i(Fields_x[:,:,-1,:], um1 = u[:,i,:], ui = u[:,i+1,:], gi = g[:,i+1,:])
             Fields_x = np.concatenate((Fields_x, new_fields), axis=2)
        #obteñamos agora as compoñentes normais dos campos
        tan_theta_r = sin_theta_r / cos_theta_r #Relaciona as compoñentes normal e tanxencial dos campos
        #redimensionamos a tanxente de theta para multiplicar elemento a elemento:
        tan_theta_r = np.stack((-tan_theta_r, tan_theta_r,-tan_theta_r, tan_theta_r), axis=1)
        Fields_z = Fields_x * tan_theta_r
        #Xuntamos as compoñentes tanxencial e normal para ter un único array Fields
        Fields = np.stack((Fields_x, Fields_z), axis=2)
        return r, t, R, T, Fields
    
    else:
        return r, t, R, T

        


#%% Funciones auxiliares

#Parte inicial do cálculo da multicapa
def Common_Matrix(ldo: float, mats: list, d: float, tempera: float , theta_i: float, Polarization: str):
    
    if type(ldo) == int or type(ldo) == float:
        ldo = [ldo]

    ldo = np.array(ldo)
    d = np.array(d)
    theta_i = np.array(theta_i).flatten()/deg
    
    index_eval = np.array([ri.refindex(ldo, mat, tempera) for mat in mats])
    index_0 = np.array([ri.refindex(ldo, mats[0], tempera) for mat in mats])

    N = theta_i.shape[0]
    M, K = index_eval.shape
    
    # Expand their dimenssions
    ds = np.expand_dims(d, axis=(0, 2))   # Shape (1, M, 1)
    thetas = np.expand_dims(theta_i, axis=(1, 2))   # Shape (N, 1, 1)
    indices_3D = np.expand_dims(index_eval, axis=0)     # Shape (1, M, K)
    ldos = np.expand_dims(ldo, axis=(0,1))     # Shape (1, 1, K)
    
    index_0 = np.expand_dims(index_0, axis=0)     # Shape (1, M, K)
    
    
    # Now broadcast x and Y to form the 3D matrices
    d_3D = np.tile(ds, (N, 1, K))         # Shape (N, M, K)
    theta_3D = np.tile(thetas, (1, M, K))         # Shape (N, M, K)
    index_3D = np.tile(indices_3D, (N, 1, 1))      # Shape (N, M, K)
    ldo_3D = np.tile(ldos, (N, M, 1))      # Shape (N, M, K)
    
    index_0 = np.tile(index_0, (N, 1, 1))      # Shape (N, M, K)
    
    sin_theta_r = np.sin(theta_3D) * index_0/index_3D
    cos_theta_r = np.sqrt((1.+0j) - sin_theta_r**2)
    
    # #segundo o tipo de polarización cambia o cálculo de índice efectivo
    if Polarization.lower() == 'tm':
        u = index_3D / cos_theta_r
    elif Polarization.lower() == 'te': 
        u = index_3D * cos_theta_r
    
    # Número de ondas no baleiro
    k0 = 2. * np.pi / ldo_3D
    # Fase adquirida ao atravesar cada unha das capas
    g = k0 * index_3D * d_3D * cos_theta_r
    
    return N, M, K, index_3D, sin_theta_r, cos_theta_r, u, g



def Create_Multilayer_Matrix(N: int, M: int, K: int, g: complex, u: complex):

    # Create a 2x2 identity matrix
    identity_matrix = np.eye(2, dtype=complex)
    Ms = expandNK(identity_matrix, N, K)
       
    # Compute the cumulative transfer matrix across layers
    for i in range(1, M - 1): 
        M_layer = Mnew(g[:, i, :], u[:, i, :])  # Shape (N, 2, 2, K)
        # Perform matrix multiplication for each angle and wavelength
        Ms = np.einsum('nijw,njkw->nikw', Ms, M_layer)  # Shape remains (N, 2, 2, K)
    
    return Ms


def Mnew(g,u):
    M00 = np.cos(g)
    M01 = -1j * np.sin(g) / u 
    M10 = -1j * np.sin(g) * u   
    M11 = np.cos(g)
    # Stack these into a 4D array and transpose to get the shape (N, 2, 2, K)
    M_new = np.array([[M00, M10], [M01, M11]])  # Shape (2, 2, N, K)
    M_new = np.transpose(M_new, (2, 1, 0, 3))   # Reorder to (N, 2, 2, K)    
    return M_new

def A(M, u0, uf):
    A00 = M[:, 0, 0] + M[:, 0, 1] * uf
    A01 = np.full_like(A00, -1.0) 
    A10 = (M[:, 1, 0] + M[:, 1, 1] * uf) / u0
    A11 = np.full_like(A00, 1.0)
  
    A = np.stack([np.stack([A00, A01], axis=-1), np.stack([A10, A11], axis=-1)], axis=1)
    return np.moveaxis(A, -1, 0)

def B(N,K):
    return np.ones([N,K,2])


#Campos -> usando c.c.:
def Campos_i(fields_m1: complex, um1: complex, ui: complex, gi: complex):    
    n,o,k = np.shape(fields_m1)
    
    Em1_p = fields_m1[:,2,:]
    Em1_m = fields_m1[:,3,:]
    
    Eil_p = 0.5 * ( (Em1_p + Em1_m) + um1/ui * (Em1_p - Em1_m) )
    Eil_m = 0.5 * ( (Em1_p + Em1_m) - um1/ui * (Em1_p - Em1_m) )
    
    Ei_p = Eil_p * np.exp(  1j * gi )
    Ei_m = Eil_m * np.exp( -1j * gi )
    
    aux1 = np.array([[1.],[0.],[0.],[0.]])
    aux2 = np.array([[0.],[1.],[0.],[0.]])
    aux3 = np.array([[0.],[0.],[1.],[0.]])
    aux4 = np.array([[0.],[0.],[0.],[1.]])
    
    new_fields = increase_dims(Eil_p) * expandNK(aux1,n,k) + increase_dims(Eil_m) * expandNK(aux2,n,k) + increase_dims(Ei_p) * expandNK(aux3,n,k) + increase_dims(Ei_m) * expandNK(aux4,n,k)

    return new_fields 

def Campos_totais(Fields):
    esquerda = Fields[:,0,:,:,:] + Fields[:,1,:,:,:] 
    dereita = Fields[:,2,:,:,:] + Fields[:,3,:,:,:] 
    return np.stack((esquerda,dereita),axis=1)


def expandNK(A, N, K):
    # Repeat the matrix along the new dimensions (N for angles, K for wavelengths)
    B = np.repeat(A[None, :, :, None], N, axis=0) # Repeat along the first axis (N)
    return np.repeat(B, K, axis=-1) # Repeat along the last axis (K)

#for fields calculation
def increase_dims(X):
    X = np.repeat(X[:, None, None, :], 4, axis=1)
    X = np.repeat(X, 1, axis=-2) 
    return X
