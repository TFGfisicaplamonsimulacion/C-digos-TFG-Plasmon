#%% Importar librerias
import cupy as cp
from typing import Optional

import os
dir_name = os.path.dirname(__file__)

#%% Referencias bibliograficas

''' Sellmeier formula [fuerzas, resonancias**2 (mum**2), rango espectral (mum)]'''

sell_res = dict(FS_Malitson =  [[0.6961663, 0.4079426, 0.8974794],
                                [0.0684043**2, 0.1162414**2, 9.896161**2],
                                [0.21, 6.7]],
                
                FS_RISBI =     [[0.9310, 0.1735, 2.1121],
                                [0.079**2, 0.130**2, 14.918**2],
                                [0.26, 1.7]],
                
                BK7 =          [[1.03961212, 0.231792344, 1.01046945],
                                [0.00600069867, 0.0200179144, 103.560653],
                                [0.3, 2.5]],
                
                F2 =           [[1.34533359, 0.209073176, 0.937357162],
                                [0.00997743871, 0.0470450767, 111.886764],
                                [0.32, 2.5]],
                
                SF11 =         [[1.73759695, 0.313747346, 1.89878101],
                                [0.013188707, 0.0623068142, 155.23629],
                                [0.37, 2.5]],
                
                SF10 =         [[1.62153902, 0.256287842, 1.64447552],
                                [0.0122241457, 0.0595736775, 147.468793],
                                [0.38, 2.5]],

                SF1 =          [[1.55912923, 0.284246288, 0.968842926],
                                [0.0121481001, 0.0534549042, 112.174809],
                                [0.2483, 2.3254]],
                
                Sapphire =     [[1.4313493, 0.65054713, 5.3414021],
                                [0.0726631**2, 0.1193242**2, 18.028251**2],
                                [0.2, 5.]])



''' Drude-Lorentz formula '''
dru_lor = dict()

''''Valores discretos'''

exp_dis = dict( Au_bulk    = './Raw_data/JohnsonAu.txt',
                Au_44nm    = './Raw_data/44nmAu.txt',
                
                Ag_bulk    = './Raw_data/Ag_Ferrera.txt',
                Ag_Wu      = './Raw_data/Ag_Wu.txt',
                Ag_20nm    = './Raw_data/Ciesielski.txt',
                
                Al_bulk    = './Raw_data/Al_Cheng.txt',
                
                Cu_bulk    = './Raw_data/Mcpeak.txt',
                
                Cr_bulk    = './Raw_data/JohnsonCr.txt',
                Cr_12nm    = './Raw_data/12nmCr_Sytchkova.txt',
                
                Si         = './Raw_data/Si_Schinke2015.txt',
                
                MgF2_44nm  = './Raw_data/44nmMgF2.txt',
                
                Water_20   = './Raw_data/Auga_Kedenburg.txt')


#%% Funcion main

def refindex(ldo: float, material: str, mod: Optional[str] = None) -> float:
    '''
    Parameters
    ----------
    ldo : float
        Vector de longitudes de onda en [nm]
    material : str
        Material del que se quiere el indice
    mod : str, optional
        Modificador espeficico como Temperatura, grosor de capa o Autor. The default is None.

    Returns
    -------
    Refractive index. (Complex)

    '''
    ldo = cp.array(ldo)
    x = ldo*1e-3
    n = cp.zeros(cp.shape(x),dtype=cp.complex_)
    
    # Comprobar si existe el material solo
    n += checkMODEL(x, n, material)

    if cp.count_nonzero(n):
        pass
    else: 
        match material.lower():
        # Casos multiples (varias referencias)
            case 'fs':
                name = material + '_' + str(mod or 'Malitson')
                n = checkMODEL(x, n, name)
            case 'au' | 'ag' | 'al' | 'cr' | 'cu':
                name = material + '_' + str(mod or 'bulk')
                n = checkMODEL(x, n, name)
            case 'water':
                name = material + '_' + str(mod or '20')
                n = checkMODEL(x, n, name)
            case 'mgf2':
                name = material + '_' + str(mod or '44nm')
                n = checkMODEL(x, n, name)
            case 'air':
                # Condiciones room*
                temperature = cp.array(20).reshape(1) # Celsius
                pressure = cp.array(101.3).reshape(1) # kPa
                humidity = cp.array(60).reshape(1) # %
                method = 'ciddor'
                xCO2 = cp.array(450).reshape(1) # ppm
                n = air_index(x*1e3,temperature , pressure, humidity, method, xCO2)
                
            case 'void':
                n += 1
            case _:
                print("Material {} not listed, check name:\n".format(material)) 
                print(list(sell_res.keys()) +
                      list(dru_lor.keys())  +
                      list(exp_dis.keys())  +
                      list(['void']), )
                return []

    return n

def list_index(materials: list):
    list_mats =[ lambda x, mat = mat: refindex(x,mat) for mat in materials]        
    return list_mats

### Check which model is avaliable
def checkMODEL(ldo, n, name):
    if name.lower() in map(str.lower, sell_res.keys()):
        matching_name = next((key for key in sell_res.keys() if key.lower() == name.lower()), None)
        n = sellmeier(ldo, sell_res[matching_name])
        
    elif name.lower() in map(str.lower, dru_lor.keys()):
        pass
    
    elif name.lower() in map(str.lower, exp_dis.keys()):
        
        matching_name = next((key for key in exp_dis.keys() if key.lower() == name.lower()), None)
        n = nk_txt(ldo,exp_dis[matching_name])
        
    return n

### Check if ldo is in biblio range
def checkLDO(ldo, ldo_range):
    if cp.max(ldo)>cp.max(ldo_range) or cp.min(ldo)<cp.min(ldo_range):
        print('\nWarning: wavelengths out of range\n')
        
### Sellmeier formula
def sellmeier(ldo, res):
    
    add = cp.zeros(cp.shape(ldo))

    for a,l2 in zip(res[0],res[1]):
       add += a*ldo**2 /(ldo**2 - l2)
       
    checkLDO(ldo,cp.array(res[2]))
    
    return cp.sqrt(1 + add)

### Valores discretos desde .txt
def nk_txt(ldo:float, name: str) -> float:
    
    with open(dir_name + name, 'r') as file:
        k = cp.array([0.,0.])
        for line in file:
            try:
                k = cp.vstack([k,[float(h) for h in (line.rsplit())]])
            except:
                pass
            
            if line.rsplit() == []:
                n = k
                k = cp.array([0.,0.])
                
    n,k = cp.interp(ldo,n[1:,0],n[1:,1]), cp.interp(ldo,k[1:,0],k[1:,1])

    n = n + 1j*k

    return n

    
    
### Aire Ciddor
def air_index(wavelength, temperature, pressure, humidity, method, xCO2):
    # AIR_INDEX Computes real index of refraction of air given
    # wavelength, temperature, pressure, humidity and CO2 concentration

    # Real index of refraction is computed using the algorithms
    # available at http://emtoolbox.nist.gov/Main/Main.asp
    # Two methods can be specified: 'Ciddor' (newer)
    # or 'Edlen' (older). The function is vectorized in
    # temperature, pressure and humidity

    # 'wavelength' is the wavelength in nanometers
    # 'temperature' is the temperature in degrees Celsius
    # 'pressure' is the pressure in kiloPascal
    # 'humidity' is the percent humidity
    # 'method' is a string indicating the method: 'edlen' or 'ciddor'
    # 'xCO2' is the CO2 concentration in ppm

    # Written by John A. Smith, CIRES, University of Colorado at Boulder

    x1, x2, x3 = cp.meshgrid(temperature, 1000 * pressure, humidity)

    S = 1e6 / wavelength**2
    T = x1 + 273.15

    # IAPWS
    K1 = 1.16705214528e3
    K2 = -7.24213167032e5
    K3 = -1.70738469401e1
    K4 = 1.20208247025e4
    K5 = -3.23255503223e6
    K6 = 1.49151086135e1
    K7 = -4.82326573616e3
    K8 = 4.05113405421e5
    K9 = -2.38555575678e-1
    K10 = 6.50175348448e2
    Omega = T + K9 / (T - K10)
    A = Omega**2 + K1 * Omega + K2
    B = K3 * Omega**2 + K4 * Omega + K5
    C = K6 * Omega**2 + K7 * Omega + K8
    X = -B + cp.sqrt(B**2 - 4 * A * C)
    psv1 = 1e6 * (2 * C / X)**4

    # Over Ice
    A1 = -13.928169
    A2 = 34.7078238
    Theta = T / 273.16
    Y = A1 * (1 - Theta**-1.5) + A2 * (1 - Theta**-1.25)
    psv2 = 611.657 * cp.exp(Y)

    psv = cp.where(x1 >= 0, psv1, psv2)
    pv = (humidity / 100) * psv

    if method == 'ciddor':

        # Convert humidity to mole fraction for Ciddor
        alpha = 1.00062
        beta = 3.14e-8
        gamma = 5.60e-7
        fpt = alpha + beta * x2 + gamma * x1**2
        xv = (x3 / 100) * fpt * psv / x2

        #Ciddor Equation
        w0, w1, w2, w3 = 295.235, 2.6422, -0.03238, 0.004028
        k0, k1, k2, k3 = 238.0185, 5792105, 57.362, 167917
        a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
        b0, b1 = 5.707e-6, -2.051e-8
        c0, c1 = 1.9898e-4, -2.376e-6
        d, e = 1.83e-11, -0.765e-8
        pR1, TR1 = 101325, 288.15
        Za = 0.9995922115
        rhovs = 0.00985938
        R, Mv = 8.314472, 0.018015
        ras = 1e-8*(k1/(k0-S)+k3/(k2-S))
        rvs = 1.022e-8*(w0+w1*S+w2*S**2+w3*S**3)
        Ma = 0.0289635+1.2011e-8*(xCO2-400)
        raxs = ras*(1+5.34e-7*(xCO2-450))
        Zm = 1-(x2/T)*(a0+a1*x1+a2*x1**2+(b0+b1*x1)*xv+ 
            (c0+c1*x1)*xv**2)+(x2/T)**2*(d+e*xv**2)
        rhoaxs = pR1*Ma/(Za*R*TR1)
        rhov = xv*x2*Mv/(Zm*R*T)
        rhoa = (1-xv)*x2*Ma/(Zm*R*T)
        nr = 1+(rhoa/rhoaxs)*raxs+(rhov/rhovs)*rvs
    else:
        # Modified Edlen Equation
        A, B, C, D = 8342.54, 2406147, 15998, 96095.43
        E, F, G = 0.601, 0.00972, 0.003661
        ns = 1+1e-8*(A+B/(130-S)+C/(38.9-S))
        X = (1+1e-8*(E-F*x1)*x2)/(1+G*x1)
        ntp = 1+x2*(ns-1)*X/D
        nr = ntp-1e-10*(292.75/T)*(3.7345-0.0401*S)*pv

    return nr.squeeze()+0j


    


