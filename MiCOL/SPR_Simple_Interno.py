# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 23:35:32 2021

@author: Alejandro Doval
"""

import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.interpolate import UnivariateSpline
import sys


#Argumentos:
#n é unha lista que contén os índices de cada capa
#d é unha lista que contén os grosores de cada capa
#ldo0 é a lonxitude de onda (no baleiro)
#theta_i é o ángulo de incidencia no primeiro medio

#para ondas TM (polarizacion ||) introducir TM=True
#para ondas TE (polarizacion L) introducir TM=False


def Multicapa(ldo0, n, d, theta_i, TM):
    
    #número de capas
    ncapas = len(n)
    
    #calculemos índices de refracción efectivos (para a polarización correspondente)
    #podemos introducilo directamente nos coeficientes de Fresnel para a incidencia normal
    #é dicir, empregándoo podemos resolver como con incidencia normal
    #para obtelos precisamos o coseno do ángulo dado pola lei de Snell en cada capa
    sin_theta_r = [np.sin(theta_i)*n[0]/x for x in n]
    cos_theta_r = [np.sqrt((1.+0j)- y*y) for y in sin_theta_r]  
        
    #segundo o tipo de polarización cambia o cálculo de índice efectivo
    if TM==True:
        u = [(n[i]/cos_theta_r[i]) for i in range(ncapas)]
    else: 
        u = [(n[i]*cos_theta_r[i]) for i in range(ncapas)]
        
    #número de ondas no baleiro
    k0 = 2. * np.pi / ldo0
    #fase adquirida ao atravesar cada unha das capas
    g = [(k0 * n[i] * d[i] * cos_theta_r[i]) for i in range(ncapas)]

    #matriz identidade para incializar o bucle
    #iremos multiplicando pola matriz correspondente a cada unha das capas
    M = np.array([[1.,0.],[0.,1.]])

    for m in range(ncapas-2):
        #matriz da capa m-esima
        M00 = np.cos(g[m+1])
        M01 = -1j * np.sin(g[m+1]) / u[m+1] 
        M10 = -1j * np.sin(g[m+1]) * u[m+1]   
        M11 = np.cos(g[m+1])
        Mnew = np.array([[M00,M01],[M10,M11]])
        #matriz global das primeiras m capas
        M = np.dot(M,Mnew)


    #Debemos despexar as incognitas, r e t (son os coeficientes de Fresnel)
    #Resolvemos o sistema AX=B
    #Matriz de coeficientes
    A00 = M[0,0] + M[0,1] * u[-1]
    A01 = -1.
    A10 = ( M[1,0] + M[1,1] * u[-1] ) / u[0]
    A11 = 1.
    #Coeficientes independentes
    B00 = 1.
    B10 = 1.
    #Definimos as matrices
    A = np.array([[A00,A01],[A10,A11]])
    B = np.array([[B00],[B10]])

    # AX=B => X = A^{-1} B
    X = np.dot(np.linalg.inv(A),B)

    t, r = X[0][0], X[1][0]

    R = np.abs(r)** 2.
    
    if TM==True:
        T = (n[-1]*cos_theta_r[0])/(n[0]*cos_theta_r[-1]) * np.abs(t)** 2.
    else:
        T = (n[-1]*cos_theta_r[-1])/(n[0]*cos_theta_r[0]) * np.abs(t)** 2.
    
    return r,t,R,T










def Multicapa_Matrix(ldo0, n, d, theta_i, TM):
    
    #número de capas
    ncapas = len(n)
    
    #calculemos índices de refracción efectivos (para a polarización correspondente)
    #podemos introducilo directamente nos coeficientes de Fresnel para a incidencia normal
    #é dicir, empregándoo podemos resolver como con incidencia normal
    #para obtelos precisamos o coseno do ángulo dado pola lei de Snell en cada capa
    
    angulos = np.dot(np.reshape(theta_i,(len(theta_i),1)),np.ones((1,len(n))))
    indices = np.dot(np.ones((len(theta_i),1)),np.reshape(n,(1,len(n))))
    ds = np.dot(np.ones((len(theta_i),1)),np.reshape(d,(1,len(d))))
    
    sin_theta_r = np.sin(angulos) * n[0]/indices
    cos_theta_r = np.sqrt((1.+0j) - sin_theta_r*sin_theta_r)
        
    #segundo o tipo de polarización cambia o cálculo de índice efectivo
    if TM==True:
        u = indices/cos_theta_r
    else: 
        u = indices * cos_theta_r
        
    #número de ondas no baleiro
    k0 = 2. * np.pi / ldo0
    #fase adquirida ao atravesar cada unha das capas
    g = k0 * indices * ds * cos_theta_r

    def Mnew(g,u):
        M00 = np.cos(g)
        M01 = -1j * np.sin(g) / u 
        M10 = -1j * np.sin(g) * u   
        M11 = np.cos(g)
        M_new = np.array([[M00,M01],[M10,M11]])
        return M_new
        
    Ms = np.ones(len(angulos))
    Ms = tuple(map(lambda x: x*np.eye(2), Ms))

    for i in range(ncapas-2):
        
        Mnews = tuple(map(lambda x,y: Mnew(x,y), g[:,i+1],u[:,i+1]))
        #Mnews = Mnew(g[:,i+1],u[:,i+1])
        
        Ms = tuple(map(lambda x, y: np.dot(x,y), Ms,Mnews))
    
    #Debemos despexar as incognitas, r e t (son os coeficientes de Fresnel)
    #Resolvemos o sistema AX=B    


    def A(M,u0,uf):
        #Matriz de coeficientes
        A00 = M[0,0] + M[0,1] * uf
        A01 = -1.
        A10 = ( M[1,0] + M[1,1] * uf ) / u0
        A11 = 1.
        #Definimos as matrices
        A = np.array([[A00,A01],[A10,A11]])
        return A
    
    def B(M):
        B = np.array([[1],[1]])
        return B
    
    A = tuple(map(lambda x,y,z: A(x,y,z), Ms, u[:,0], u[:,-1]))
    B = tuple(map(lambda x: B(x), Ms))
    
    # AX=B => X = A^{-1} B
    X = tuple(map(lambda x,y: np.dot(np.linalg.inv(x),y), A,B))

    t = np.array(list(map(lambda x: x[0][0], X)))
    r = np.array(list(map(lambda x: x[1][0], X)))

    R = np.abs(r)** 2.
    
    if TM==True:
        T = (indices[:,-1]*cos_theta_r[:,0])/(indices[:,0]*cos_theta_r[:,-1]) * np.abs(t)** 2.
    else:
        T = (indices[:,-1]*cos_theta_r[:,-1])/(indices[:,0]*cos_theta_r[:,0]) * np.abs(t)** 2.
    
    return r,t,R,T





#%%



##Só ángulos internos, non consideramos a incidencia nos prismas

def int_angulo_i(ldo, n, d, res=0.1, T_calc=False, r_calc=False, t_calc=False):
    
    #Traballo SÓ con ángulos internos
    incidencia = np.arange(0,(np.pi/2.-1e-4),(res*np.pi/180))
    deg=180/np.pi
    
    #incidencia = np.arange(40/deg,55/deg,(res/deg))
    N=len(incidencia)
    #non poño ata pi/2 porque se anula o coseno e da problemas ao dividir entre u en TM

    #número de medios
    if len(n) != len(d):
        print('O número de índices e de espesores indicados non coinciden!')
        sys.exit()
              
    RsTM = []
    RsTE = []
    if T_calc == True:
        TsTM = []
        TsTE = []
    if r_calc == True:
        rsTM = []
        rsTE = []
    if t_calc == True:
        tsTM = []
        tsTE = []

    for i in range(N):
        r,t,R,T = Multicapa(ldo, n, d, incidencia[i], TM=False)
        RsTE.append(R)
        if T_calc == True:
            TsTE.append(T)
        if r_calc == True:    
            rsTE.append(r)
        if t_calc == True:    
            tsTE.append(t)
 
    for i in range(N):
        r,t,R,T = Multicapa(ldo, n, d, incidencia[i], TM=True)
        RsTM.append(R)
        if T_calc == True:
            TsTM.append(T)
        if r_calc == True:
            rsTM.append(r)
        if t_calc == True:
            tsTM.append(t)
    
    if T_calc == True:
        return incidencia, np.array(RsTM).real, np.array(RsTE).real, np.array(TsTM).real, np.array(TsTE).real   
    else:
        return incidencia, np.array(RsTM).real, np.array(RsTE).real    





##Só ángulos internos, non consideramos a incidencia nos prismas

def int_angulo_i_Matrix(ldo, n, d, incidencia='aa', res=1/60, T_calc=False, r_calc=False, t_calc=False):
    
    #Traballo SÓ con ángulos internos
    if type(incidencia)==str:
        incidencia = np.arange(0,(np.pi/2.-1e-4),(res*np.pi/180))
    # deg=180/np.pi
    # #incidencia = np.arange(40/deg,55/deg,(res/deg))
    # N=len(incidencia)
    #non poño ata pi/2 porque se anula o coseno e da problemas ao dividir entre u en TM

    #número de medios
    if len(n) != len(d):
        print('O número de índices e de espesores indicados non coinciden!')
        sys.exit()
              
    rsTE,tsTE,RsTE,TsTE = Multicapa_Matrix(ldo, n, d, incidencia, TM=False)
    rsTM,tsTM,RsTM,TsTM = Multicapa_Matrix(ldo, n, d, incidencia, TM=True)
        
    if T_calc == True:
        return incidencia, np.array(RsTM).real, np.array(RsTE).real, np.array(TsTM).real, np.array(TsTE).real   
    else:
        return incidencia, np.array(RsTM).real, np.array(RsTE).real    




#%%









def int_ldo_i(incidencia, nf, d, rango=[500,1500], res=0.1, T_calc=False, r_calc=False, t_calc=False):
#incidencia é o ángulo de incidencia interno sobre a multicapa
#n e d son as listas coas funcións dos índices e os espesores de cada un dos medios
#rango é o intervalo de lonxitudes de onda para a interrogación espectral e
#res é a resolución da interrogación espectral, ambos en nanómetros
#T_calc, r_calc, t_calc ofrecen a posibilidade de que se calculen os coeficientes
#de fresnel ou a transmitancia

    deg = 180/np.pi
    incidencia = incidencia/deg

    #array de lonxitudes de onda
    ldo0 = np.arange(rango[0],rango[-1],res)
    N=len(ldo0)
        
    #número de medios
    if len(nf) != len(d):
        print('O número de índices e de espesores indicados non coinciden!')
        sys.exit()
    n_medios = len(nf)
    
    RsTM = []
    RsTE = []
    if T_calc == True:
        TsTM = []
        TsTE = []
    if r_calc == True:
        rsTM = []
        rsTE = []
    if t_calc == True:
        tsTM = []
        tsTE = []

    for i in range(N):
        
        ns = [index(ldo0[i]) for index in nf]
        
        #TE
        r,t,R,T = Multicapa(ldo0[i], ns, d, incidencia, TM=False)
        RsTE.append(R)
        if T_calc == True:
            TsTE.append(T)
        if r_calc == True:    
            rsTE.append(r)
        if t_calc == True:    
            tsTE.append(t)

        #TM
        r,t,R,T = Multicapa(ldo0[i], ns, d, incidencia, TM=True)
        RsTM.append(R)
        if T_calc == True:
            TsTM.append(T)
        if r_calc == True:
            rsTM.append(r)
        if t_calc == True:
            tsTM.append(t)
    
    if T_calc == True:
        return ldo0, np.array(RsTM).real, np.array(RsTE).real, np.array(TsTM).real, np.array(TsTE).real   
    else:
        return ldo0, np.array(RsTM).real, np.array(RsTE).real
    


