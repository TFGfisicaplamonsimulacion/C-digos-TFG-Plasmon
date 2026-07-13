import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
####################################
# https://en.wikipedia.org/wiki/Thermal_diffusivity
####################################
#para justificar en el documento buscar coheficiente termooptico
#PRIMER DRAFT EL 10 CON GRAFICAS Y COSAS QUE QUIERO PRESENTAR
#ordenes de energia: el laser es nkt supercontimuun laser
#HACER CORRELACIÓN 
#SEMIANCHURA DEL LASER 1 MM 

#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,50,23.38,0.005,0.5 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dx  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer





red=30 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon


#maxi=np.argmin(np.abs(angulos-49))


plt.show()
plt.close()





### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)





# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N)]
A = diags(diagonals, [0, -1, 1, -N, N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N)]
M = diags(diagonals, [0, -1, 1, -N, N], format = "csr")

anch=10



tiempo=[]
ptocentr=[]
listagif,listagif2,frames= [],[],int(Nt/50)
for n in range(0,Nt):
    print(n)

    #el laser tiene un valor de 110 e-3 J/s 
    tempmed=[]
    for i in range(int(N/2-1),int(N/2+2)):
        for j in range(int(N/2-1),int(N/2+2)):
            tempmed.append(placa[i,j])

    tempera=np.mean(tempmed)+27



    placa=placa.flatten()  
    placa, _ = cgs(A,M@placa) #solo un longitud q supongo 1 nm anchura del laser de 1 nm
    placa=placa.reshape((N,N))
    tiemp=n*dt
    tiempo.append(tiemp)
    ptocentr.append(placa[1,int(N/2)])
    print(placa[int(1),int(N/2)])
    #placa[N-1,:]=0
    placa[0,:]=10

    if int(n%frames)==0:

        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="6" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()

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
        plt.title(f'Ángulo ={tiemp} θ \n Tiempo = {tiemp} s')
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()
        img2 = Image.open(buf2)
        listagif2.append(img2)
        plt.show()
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
plt.title("Evolución de la temperatura a lo largo del tiempo")
plt.xlabel("tiempo (s)")
plt.ylabel("${\Delta}T$")
curva=plt.plot(tiempo,ptocentr)
plt.savefig("caloragua")
plt.show()