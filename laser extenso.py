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


#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,100,127,0.01,3 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dy  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer


tempera=27


red=9 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon
ldo,angulos,materiales,grosores = np.arange(480.,780.,1),np.arange(0.,90,90/Nt/red) ,['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm ldo y grosores
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, tempera=tempera,mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
maxi=np.argmin(np.abs(angulos-49))
maxi=np.argmin(R)




cl=maxi*90/Abs.size  #centro del angulo en torno al que movemos el laser
cl=75
print("tenemos el plasmons")
#hacer en agua y asi ver comparativa de como cambia R en el inicial frente a medida que se va calentando, en cada punto calculo el r de ese punto y construyo puntito a puntito la nueva curva de r para ver la diferencia respecto a la que tenia de base y asi veo como se ve perturbada la curva original
#explicar en el tfg lo de que tomo 0.05 para la energia y poner loque da el proveedor del espectro
plt.figure()

n_curvas = Abs.shape[1]

# Generamos n colores del colormap 'rainbow'
colors = plt.cm.rainbow(np.linspace(0, 1, n_curvas))


# Dibujamos cada curva con su color
for i in range(n_curvas):
    plt.plot(angulos, Abs[:,i], color=colors[i], label=f'Curva {i}')
    
plt.xlabel('Ángulo (grados)')
plt.ylabel('Absorbancia')
plt.grid(False)
plt.show()





### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)




# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N),-r1*np.ones(N),-r1*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N),r1*np.ones(N),r1*np.ones(N)]
M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")


anch=1 #usamos un filtro para tener solo un ancho espectral de 3 nm por ejemplo
tiempo=[]
ptocentr=[]
plasdef=[]
angula=[]
listagif,listagif2,frames= [],[],int(Nt/100)
for n in range(0,Nt):
    ang=round((n*90/Nt-45)/red+cl,3)
    #ang=(maxi)*90/len(Abs) 
    excx=(N/10*0.4246*np.cos(ang*2*np.pi/360))**2

    excy=(N/10*0.4246)**2
    laser = np.exp(-((X-cordX)**2/(2*excx) + (Y-cordY)**2/(2*excy)))# Perfil gaussiano 
    laser=laser/np.sum(laser)

    #para anchos espectreales sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])) sin el anch pq ya lo tengo, para eso hacer que me de una matrix el programa con un lado ang y otro lambda
    
    placa=placa+laser*sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])*0.06*dt/2.492/(np.pi*1e-4)#LASER EN MOVIMIENTO
    print(laser[int(N/2),int(N/2)]*sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])*0.06*dt/2.492/(np.pi*1e-4))
    #placa=placa+laser*(Abs)[maxi]*0.06*anch*dt/2.492/(np.pi*1e-4) #LASER QUIETO EN UN PUNTO
    placa=placa.flatten()  
    placa, _ = cgs(A,M@placa) 
    placa=placa.reshape((N,N))
    angula.append(ang)
    tiemp=n*dt
    tiempo.append(tiemp)
    ptocentr.append(placa[int(N/2),int(N/2)])
    placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0
    
    tempmed=[]
    for i in range(int(N/2-1),int(N/2+2)):
        for j in range(int(N/2-1),int(N/2+2)):
            tempmed.append(placa[i,j])

    tempera=np.mean(tempmed)+27
    
    r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
    Absdef= (1 - R[int(n+len(Abs)*(cl-45/red)/90)] -T[int(n+len(Abs)*(cl-45/red)/90)].real)
    plasdef.append(Absdef)
    print(n, "de", Nt)
    if int(n%frames)==0:
        tiemp=round(n*dt,3)
  
        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="10" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
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
        ax2.set_zlabel("Grados")
        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()
        img2 = Image.open(buf2)
        listagif2.append(img2)


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


vang=round(np.pi*(angula[Nt-1]-angula[0])/Nt/180*1000,4)
plt.figure()
plt.title("Evolución de la temperatura a lo largo del tiempo \n Plata y agua")
plt.xlabel("Tiempo (s)")
plt.ylabel("${\Delta}T$")
curva=plt.plot(tiempo,ptocentr)
plt.ylim(0,25)

plt.savefig("curva mov")
plt.show()




print( "TERMINAMOS!!")