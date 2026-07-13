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
#ordenes de energia: el laser es nkt supercontimuun laser DE AHI TOMAR LOS VALORES ALGUN DIA HABRÁ QUE PONER UNIDADES 
#HACER CORRELACIÓN 
#SEMIANCHURA DEL LASER 1 MM 

#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,50,127,0.05,30 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dx  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer, pongo temperatura ambiente
placa=placa.flatten()



### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)


# parametros del plasmon inicial (suponemos temperatura ambiente)
tempera = 300
ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt), ['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, mats= materiales,tempera=tempera,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
plt.figure()
plt.plot(angulos,Abs)
plt.show()


for i in range(len(Abs)):
    if Abs[i]==max(Abs):
        maxi=i
ang=maxi*90/len(Abs) #el angulo así está fijado en el punto más caliente del plasmon en el estado de reposo

# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N),-r1*np.ones(N),-r1*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N),r1*np.ones(N),r1*np.ones(N)]
M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")


listagif,listagif2= [],[]

for n in range(1,Nt):
    
    excx=(0.4246*np.cos(ang))**2
    excy=(0.4246)**2
    laser = np.exp(-((X-cordX)**2/excx + (Y-cordY)**2/excy))/(0.5*np.pi*np.sqrt(excx)*np.sqrt(excy))# Perfil gaussiano 
    tempmed=[]
    for i in range(int(N/2-1),int(N/2+2)):
        for j in range(int(N/2-1),int(N/2+2)):
            tempmed.append(placa[i*N+j])
        
    ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt), ['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm
    r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, mats= materiales,d= grosores, tempera= tempera,theta_i= angulos)
    Abs = (1 - R -T.real)


    placa, _ = cgs(A,M@placa + laser.flatten()*(Abs)[maxi]*0.110*dt/2.492/4e-14)
    #poner linespace
    placa=placa.reshape((N,N))
    placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0
    #unidades, refindex usa kelvin

    if n%20==0:
        tiemp=n*dt
        
        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="3" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()
        plt.title(f'Estamos en el ángulo:{ang}')
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        img = Image.open(buf)
        listagif.append(img)

        ### CREAMOS LA IMAGEN 3D Y LA GUARDAMOS 
        fig2 = plt.figure()
        ax2 = plt.axes(projection="3d")
        ax2.plot_surface(X, Y,placa, cmap="plasma",edgecolor="none") 
        ax2.set_zlim(0)

        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()
        img2 = Image.open(buf2)
        listagif2.append(img2)
        print("Llevamos", n ," ciclos, es decir:\nLlevamos",n*dt,"segundos")
        
    placa=placa.flatten()     
    
    
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