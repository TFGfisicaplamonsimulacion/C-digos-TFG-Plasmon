# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:33:06 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import time
import os

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela (1)\\Instrumentación_Laboratorio\\Simulacion')
from Indices import Index as ri
from MiCOL import Main_SPR as cr


#%% definimos a función de axuste do espesor:
def Axustar_espesor(mats, alpha, input_espesores, input_extremos, cociente_brazos, carpeta, file_name, plt_show):

    dM1, dM2 = input_espesores[0], input_espesores[1]
    dC1, dC2 = input_espesores[2], input_espesores[3]
    
    ldo_min, ldo_max = input_extremos[0], input_extremos[1]
    theta_min, theta_max = input_extremos[2], input_extremos[3]
    ldo_min_adx, ldo_max_adx = input_extremos[4], input_extremos[5]
    theta_min_adx, theta_max_adx = input_extremos[6], input_extremos[7]
    
    show_ext, show_Texp, show_correl, show_Tteo = plt_show[0], plt_show[1], plt_show[2], plt_show[3]


    #%%cargamos a medida
    #imos á carpeta onde está/n a/s medida/s:
    os.chdir(carpeta)
    
    #cargamos a medida, almacenando valores, lonxitudes de onda e ángulos
    df = pd.read_csv(file_name + '.csv', index_col='Varphi')
    matriz_T = df.values
    ldos = df.columns.astype(float)
    varphis = df.index[4:]
    
    #definimos o rango de lonxitudes de onda e angulos a usar:
    rng_ldo = ( ldos > ldo_min ) * ( ldos < ldo_max )
    
    #Lemos os darks tomados cos t_exp da referencia e da medida
    dark_ref, dark_medida = matriz_T[1,:], matriz_T[2,:]
    #lemos a referencia e restámoslle o seu dark, e multiplicamos polo cociente entre os brazos
    referencia = (matriz_T[0,:] - dark_ref) * cociente_brazos
    #restamos a cada fila o dark e dividimos o resultado entre a medida de referencia
    matriz = ( matriz_T[4:,:] - np.meshgrid(dark_medida,varphis)[0] ) 
    medida = matriz / np.meshgrid(referencia,varphis)[0]
    #poñemos a cero todos os valores negativos da matriz de medida que temos como resultado (ruído)
    medida[medida<0] = 0

    #%% representación en ángulos externos
    
    #transformamos a externos o rango de ángulos internos
    varphi_min = cr.theta2varphi(theta_int = theta_min, ldo = 800, prism = mats[0], alpha = alpha)
    varphi_max = cr.theta2varphi(theta_int = theta_max, ldo = 800, prism = mats[0], alpha = alpha)
    if np.isnan(varphi_min):
        varphi_min = -90
    if np.isnan(varphi_max):
        varphi_max = 90
    rng_varphis = ( varphis > varphi_min ) * ( varphis < varphi_max )
    
    if show_ext == True:
        plt.figure(11)
        plt.clf()
        plt.pcolormesh(ldos[rng_ldo], varphis[rng_varphis], medida[rng_varphis][:,rng_ldo], cmap='turbo')
        plt.xlabel(r'$\lambda\ (nm)$', size=22)
        plt.ylabel(r'$\varphi\ (^{\circ})$', size=22)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('T', size=22,rotation=0, loc='top')



    #%%transformamos a ángulos internos:
        
    #para cambiar a ángulos internos traballaremos por separado a cada lonxitude de onda fixa
    #para cada lonxitude de onda, almacenamos o theta mínimo e o máximo
    #o noso rango de thetas irá dende o maior dos mínimos ata o máis pequeno dos máximos
    min_theta = np.array([])
    max_theta = np.array([])
    Y_interp = np.zeros((len(ldos),len(varphis)))
    for i in range(len(ldos)):   
        thetas_i = cr.varphi2theta(var_phi = varphis, ldo = ldos[i], prism = mats[0], alpha = alpha)
        Y_interp[i,:] = thetas_i
        min_theta = np.append(min_theta, np.min(thetas_i))
        max_theta = np.append(max_theta, np.max(thetas_i))
    theta_minim = np.max(min_theta)
    theta_maxim = np.min(max_theta)    
    res_thetas = np.max(np.abs(np.diff(varphis)))
    thetas = np.arange(theta_minim, theta_maxim, res_thetas)    
        
    Matriz_valores = np.zeros((len(thetas),len(ldos)))
    for i in range(len(ldos)):
        thetas_i = Y_interp[i,:]
        values_i = medida[:,i]
        f_interp = interp1d(thetas_i,values_i,kind='cubic')
        valores = f_interp(thetas)
        Matriz_valores[:,i] = valores    
    
    Matriz_valores[Matriz_valores<0] = 0
    T_exp = Matriz_valores
    
    #definimos o rango de ángulos a empregar
    rng_ang = ( thetas > theta_min ) * ( thetas < theta_max )



    #%% representamos a medida en ángulos internos
    
    if show_Texp == True:
        plt.figure(12)
        plt.clf()
        plt.pcolormesh(ldos[rng_ldo], thetas[rng_ang], T_exp[rng_ang][:,rng_ldo], cmap='turbo')
        plt.xlabel(r'$\lambda\ (nm)$', size=22)
        plt.ylabel(r'$\theta\ (^{\circ})$', size=22)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('T', size=22,rotation=0, loc='top')
        
    
    
    
    #%% Se queremos usar para o axuste unicamente os rangos seleccionados, cortamos aquí:
    ldos = ldos[rng_ldo]
    thetas = thetas[rng_ang]
    T_exp = T_exp[rng_ang][:,rng_ldo]
    
    
    
    #%% Estimación do espesor usando toda a matriz de datos dispoñibles
    rng_ldo_adx = (ldos > ldo_min_adx) * (ldos < ldo_max_adx)
    rng_thetas_adx = (thetas > theta_min_adx) * (thetas < theta_max_adx)
    ldos_adx = ldos[rng_ldo_adx]
    thetas_adx = thetas[rng_thetas_adx]
    T_adx = T_exp[rng_thetas_adx][:,rng_ldo_adx]
    
    d_scan = np.arange(300, 5000, 100)
    
    for iteration in [1,2]: 
        correls = []
        for dD in d_scan:
            d = [0, dM1, dC1, dD, dC2, dM2, 0]
            r, t, R, T = cr.Multicapa_Matrix(ldo = ldos_adx, mats = mats, d = d, theta_i = thetas_adx, Polarization = 'tm', Campos = False)
            correl = ( np.sum(T_adx * T.real) )**2 / ( np.sum(T_adx * T_adx) * np.sum(T.real * T.real) )
            correls.append(correl)
        
        k_max = np.argmax(correls)
        dD_estimado = d_scan[k_max]
        d_scan = d_scan = np.arange(dD_estimado-100, dD_estimado+100, 10)
        
    if show_correl == True:
        plt.figure(16)
        plt.clf()
        plt.plot(d_scan,np.abs(correls))
        plt.xlabel(r'$d_{scan}$ (nm)', size=18)
        plt.title('Correlacion Mapa Completo', size=22)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
    final_d = dD_estimado
    print('Estimación: d= %.1f nm' %(final_d))
    return final_d
    
    d = [0, dM1, dC1, final_d, dC2, dM2, 0]
    r, t, R, T = cr.Multicapa_Matrix(ldo = ldos, mats = mats, d = d, theta_i = thetas, Polarization = 'tm', Campos = False)
    
    if show_Tteo == True:
        plt.figure(22)
        plt.clf()
        plt.pcolormesh(ldos, thetas, T.real, cmap='turbo')
        plt.xlabel(r'$\lambda\ (nm)$', size=22)
        plt.ylabel(r'$\theta\ (^{\circ})$', size=22)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(r'$T_{teo}$', size=22,rotation=0, loc='top')
        
        
        
        
        
        

#no caso isósceles (asumindo que a diferenza entre os alphas dunha e doutra cara é pequena):
    
def InOut_Kr(varphi: float, ldo: float, prism: str, Polarization = 'tm'):    
    
    nair = ri.refindex(ldo, material = 'air')
    nprism = ri.refindex(ldo, material = prism)
    
    deg = 180/np.pi
    varphi = varphi/deg
    phi = np.arcsin(nair * np.sin(varphi) / nprism)
    
    if Polarization == 'tm':
        uair = nair/np.cos(varphi)
        uprism = nprism/np.cos(phi)
        
    elif Polarization == 'te':
        uair = nair * np.cos(varphi)
        uprism = nprism * np.cos(phi)
    
    r_in = (uair - uprism) / (uair + uprism)
    T_in = 1 - np.abs(r_in)**2
    T = T_in * T_in
    
    return T
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        