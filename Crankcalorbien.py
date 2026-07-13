import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
from MiCOL import Main_SPR_no_va as cr
from Indices import Index_no_va as ri
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
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dx  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer





red=30 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon
ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt/red) ,['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm ldo y grosores
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=27, mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
maxi=np.argmin(R)

#maxi=np.argmin(np.abs(angulos-49))

plt.plot(angulos,Abs)
plt.show()
plt.close()





### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)





# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N),-r1*np.ones(N),-r1*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N),r1*np.ones(N),r1*np.ones(N)]
M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")

anch=5

resos=[]
ang=maxi*90/len(Abs)
tiempo=[]
ptocentr=[]
listagif,listagif2,frames= [],[],int(Nt/40)
for n in range(0,Nt):
    
    excx=(N/10*0.4246*np.cos(ang*2*np.pi/360))**2
    excy=(N/10*0.4246)**2
    laser = np.exp(-((X-cordX)**2/(2*excx) + (Y-cordY)**2/(2*excy)))# Perfil gaussiano no normalizado
    laser=laser/np.sum(laser) #Asi lo normalizo sin problemas numericos
    #el laser tiene un valor de 110 e-3 J/s 
    tempmed=[]
    for i in range(int(N/2-9),int(N/2+10)):
        for j in range(int(N/2-9),int(N/2+10)):
            tempmed.append(placa[i,j])

    tempera=np.mean(tempmed)/2+27

    r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, mats= materiales,d= grosores, tempera= tempera,theta_i= angulos) #puedo poner un angulo solo y poner un vector con las longitudes de onda,1 de distancia pq si no el 0.05 no fnciona, y asi integro eso y puedo tener en cuenta un ancho espectral
    Abs = (1 - R -T.real)
    resos.append(np.argmin(R)*90/len(R))

    #print(ri.water_index(ldo,tempera), "tempera=",tempera)
    placa=placa+laser*(Abs)[maxi]*0.05*anch*dt/2.492/1e-4
    placa=placa.flatten()  
    placa, _ = cgs(A,M@placa) #solo un longitud q supongo 1 nm anchura del laser de 1 nm
    placa=placa.reshape((N,N))
    tiemp=n*dt
    tiempo.append(tiemp)
    ptocentr.append(placa[int(N/2),int(N/2)])
    placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0

    if int(n%frames)==0:

        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="6" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        #plt.show()
        plt.close()
        img = Image.open(buf)
        listagif.append(img)

        ### CREAMOS LA IMAGEN 3D Y LA GUARDAMOS 
        fig2 = plt.figure()
        ax2 = plt.axes(projection="3d")
        ax2.plot_surface(X, Y,placa, cmap="plasma",edgecolor="none") 
        ax2.set_zlim(0,3)

        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()
        img2 = Image.open(buf2)
        listagif2.append(img2)

        plt.plot(angulos,Abs)
        #plt.show()
        plt.close()
   
    
    
listagif[0].save(
    "output.gif",
    save_all=True,        # Save multiple frames
    append_images=listagif[1:],  # Add remaining frames
    duration=200,         # Frame duration in milliseconds
    loop=0)             
listagif2[0].save(
    "output2.gif",
    save_all=True,        # Save multiple frames
    append_images=listagif2[1:],  # Add remaining frames
    duration=200,         # Frame duration in milliseconds
    loop=0)  

plt.figure()
plt.title("Evolución del angulo crítico a lo largo del tiempo")
plt.xlabel("tiempo (s)")
plt.ylabel("${\Delta}{\Theta}$")
curva=plt.plot(tiempo,resos-maxi*90/len(Abs))
plt.savefig("curva critico")
plt.show()

plt.figure()
plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Ángulo ={ang} mRad/s ")
plt.xlabel("tiempo (s)")
plt.ylabel("${\Delta}$")
curva=plt.plot(tiempo,ptocentr)
plt.savefig("curva calor")
plt.show()