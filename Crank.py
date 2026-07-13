import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
####################################

#####################################
# https://en.wikipedia.org/wiki/Thermal_diffusivity
#Forma de la placa y parametros

L= 50 # Tamaño
N = 50  # Puntos
dx, dy = L / (N - 1), L / (N - 1)  # Separaciones
alpha = 127 #alfa del oro para plata es 165.63
dt = 0.5
T_final = 1000  # Tiempo total
Nt = int(T_final / dt)  # Numero de pasos
r=alpha*dt/2/dx/dx
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer, pongo temperatura ambiente
placa=placa.flatten()
print(placa)

#######################################################################
# crank nicolson mirar ese método xDDDDDDDDDDDDDDDDDD




for i in range(1):
    ### Parametros del laser
    cordX=L/2
    cordY=L/2
    excx=100
    excy=1
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    laser = np.exp(-((X-cordX)**2/excx + (Y-cordY)**2/excy))/0.1# Perfil gaussiano 

# Matrices de mi sistema
diagonals = [(1 + 4*r)* np.ones(N*N), -r*np.ones(N*N - 1), -r*np.ones(N*N -1), -r*np.ones(N*N - N), -r*np.ones(N*N - N),-r*np.ones(N),-r*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")

diagonals = [(1-4*r)* np.ones(N*N), r*np.ones(N*N - 1), r*np.ones(N*N -1), r*np.ones(N*N - N), r*np.ones(N*N - N),r*np.ones(N),r*np.ones(N)]


M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")


listagif= []
listagif2 =[]
for n in range(1,Nt):
    placa, _ = cgs(A,M@placa + laser.flatten()*(1+n/100))
    placa=placa.reshape((N,N))
    placa[:,N-1]=0
    placa[N-1,:]=0
    placa[:,0]=0
    placa[0,:]=0

    if n%20==0:
        
        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="30" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()

        
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()  # Free memory

        img = Image.open(buf)
        listagif.append(img)


        

        
        ### CREAMOS LA IMAGEN 3D Y LA GUARDAMOS
        
        fig2 = plt.figure()
        ax2 = plt.axes(projection="3d")
        ax2.plot_surface(X, Y,placa, cmap="plasma",edgecolor="none") 
        ax2.set_zlim(np.min(placa), 30)
        plt.gca().invert_yaxis()

        

        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()  # Free memory

        img2 = Image.open(buf2)
        listagif2.append(img2)
        print("Llevamos", n ," ciclos, es decir:\nLlevamos",n*dt,"segundos")
        
    placa=placa.flatten()     
    
    
listagif[0].save(
    "output.gif",
    save_all=True,        # Save multiple frames
    append_images=listagif[1:],  # Add remaining frames
    duration=200,         # Frame duration in milliseconds
    loop=0                # 0 = Loop forever, set to 1 for no loop
)             
listagif2[0].save(
    "output2.gif",
    save_all=True,        # Save multiple frames
    append_images=listagif2[1:],  # Add remaining frames
    duration=200,         # Frame duration in milliseconds
    loop=0                # 0 = Loop forever, set to 1 for no loop
)  