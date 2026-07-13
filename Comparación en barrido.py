import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
from MiCOL import Main_SPR_no_va as cr
####################################
# https://en.wikipedia.org/wiki/Thermal_diffusivity
####################################
#para justificar en el documento buscar coheficiente termooptico
#PRIMER DRAFT EL 10 CON GRAFICAS Y COSAS QUE QUIERO PRESENTAR
#ordenes de energia: el laser es nkt supercontimuun laser
#HACER CORRELACIÓN 
#SEMIANCHURA DEL LASER 1 MM 

#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,100,127,0.05,30 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dy  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer


tempera=27


red=30 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon
ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt/red) ,['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm ldo y grosores
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, tempera=tempera,mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
maxi=np.argmin(np.abs(angulos-49))
maxi=np.argmin(R)
cl=maxi*90/len(Abs)  #centro del angulo en torno al que movemos el laser


#hacer en agua y asi ver comparativa de como cambia R en el inicial frente a medida que se va calentando, en cada punto calculo el r de ese punto y construyo puntito a puntito la nueva curva de r para ver la diferencia respecto a la que tenia de base y asi veo como se ve perturbada la curva original
#explicar en el tfg lo de que tomo 0.05 para la energia y poner loque da el proveedor del espectro
plt.figure()
plt.plot(angulos,R)
plt.show()





### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)


#CORREGIR EN EL OTRO PROGRAMA LO DE LA PLACA TAMAÑO


# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N),-r1*np.ones(N),-r1*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N),r1*np.ones(N),r1*np.ones(N)]
M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")


anch=15 #usamos un filtro para tener solo un ancho espectral de 3 nm
tiempo=[]
ptocentr=[]
plasdef=[]
listagif,listagif2,frames= [],[],int(Nt/100)
for n in range(0,Nt):
    ang=round((n*90/Nt-45)/red+cl,3)
    #ang=(maxi)*90/len(Abs) 
    excx=(N/10*0.4246*np.cos(ang*2*np.pi/360))**2

    excy=(N/10*0.4246)**2
    laser = np.exp(-((X-cordX)**2/(2*excx) + (Y-cordY)**2/(2*excy)))# Perfil gaussiano 
    laser=laser/np.sum(laser)

    #para anchos espectreales sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])) sin el anch pq ya lo tengo, para eso hacer que me de una matrix el programa con un lado ang y otro lambda
    
    placa=placa+laser*(Abs)[int(n+len(Abs)*(cl-45/red)/90)]*0.06*anch*dt/2.492/(1e-4)#LASER EN MOVIMIENTO
    #placa=placa+laser*(Abs)[maxi]*0.06*anch*dt/2.492/(np.pi*1e-4) #LASER QUIETO EN UN PUNTO
    placa=placa.flatten()  
    placa, _ = cgs(A,M@placa) 
    placa=placa.reshape((N,N))

    tiemp=n*dt
    tiempo.append(tiemp)
    ptocentr.append(placa[int(N/2),int(N/2)])
    placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0
    
    tempmed=[]
    for i in range(int(N/2-1),int(N/2+2)):
        for j in range(int(N/2-1),int(N/2+2)):
            tempmed.append(placa[i,j])

    tempera=np.mean(tempmed)*0.7+27
    
    r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
    Absdef= (1 - R[int(n+len(Abs)*(cl-45/red)/90)] -T[int(n+len(Abs)*(cl-45/red)/90)].real)
    plasdef.append(Absdef)

    if int(n%frames)==0:
        tiemp=round(n*dt,3)
        print(n)

   
    
    



plt.figure()
plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],np.array(plasdef),label="Ajuste según la temperatura")
plt.plot(angulos,Abs,label="Valor inicial de la curva")
plt.title("Cambio de la curva de resonancia ")
plt.legend()
plt.xlim(52,56)
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("cambio abs")
plt.show()




plt.figure()
plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Ángulo ={ang} θ ")
plt.xlabel("tiempo (s)")
plt.ylabel("${\Delta}T$")
curva=plt.plot(tiempo,ptocentr)
plt.ylim(0,8)
plt.savefig("curva mov")
plt.show()

plt.figure()

plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],Abs[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))]-np.array(plasdef))
plt.xlabel("tiempo (s)")
plt.ylabel("Diferencia de absorción")
plt.savefig("Diferencia ")
plt.show()