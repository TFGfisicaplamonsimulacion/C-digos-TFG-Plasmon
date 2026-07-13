import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
from MiCOL import Main_SPR_no_va as cr



#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,100,127,0.005,50 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dy  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer


tempera=27 #Temperatura ambiente en celsius


red=30 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon
ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt/red) ,['BK7','au', 'air' ], [ 0., 40. ,0. ] # en nm ldo y grosores
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, tempera=tempera,mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real) #Curva de absorcion




maxi=np.argmin(R)
#maxi=np.argmin(abs(angulos-54))
cl=maxi*90/len(Abs)  #centro del angulo en torno al que movemos el laser
cl=cl+1
plt.figure()
plt.plot(angulos,R)
plt.show()


### Parametros del laser
cordX,cordY,x,y=L/2,L/2,np.linspace(0, L, N),np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)





# Matrices de mi sistema
diagonals = [(1 + 4*r1)* np.ones(N*N), -r1*np.ones(N*N - 1), -r1*np.ones(N*N -1), -r1*np.ones(N*N - N), -r1*np.ones(N*N - N),-r1*np.ones(N),-r1*np.ones(N)]
A = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format="csr")
diagonals = [(1-4*r1)* np.ones(N*N), r1*np.ones(N*N - 1), r1*np.ones(N*N -1), r1*np.ones(N*N - N), r1*np.ones(N*N - N),r1*np.ones(N),r1*np.ones(N)]
M = diags(diagonals, [0, -1, 1, -N, N,N*N-N, -N*N+N], format = "csr")


anch=9.8 #usamos un filtro para tener solo un ancho espectral de 10 nm por ejemplo
maximociclo=[]
maximociclopos=[]
tiempo=[]
ptocentr=[]
plasdef=[]
angula=[]
listagif=[]
listagif2=[]
frames=int(Nt/10)

for j in range(4):   
    for n in range(0,Nt):
        ang=round((n*90/Nt-45)/red+cl,3)
        #ang=(maxi)*90/len(Abs)
        
        #Laser ajustandose en cada paso
        excx=(N/10*0.4246*np.cos(ang*2*np.pi/360))**2
        excy=(N/10*0.4246)**2
        laser = np.exp(-((X-cordX)**2/(2*excx) + (Y-cordY)**2/(2*excy)))# Perfil gaussiano 
        laser=laser/np.sum(laser)
    
        #para anchos espectreales sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])
        #Operaciones 
        placa=placa+laser*(Abs)[int(n+len(Abs)*(cl-45/red)/90)]*0.06*anch*dt/2.492/(1e-4)#LASER EN MOVIMIENTO
        #placa=placa+laser*(Abs)[maxi]*0.06*anch*dt/2.492/(np.pi*1e-4) #LASER QUIETO EN UN PUNTO
        placa=placa.flatten()  
        placa, _ = cgs(A,M@placa) 
        placa=placa.reshape((N,N))
        placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0 #Condiciones de contorno
        
        
        angula.append(ang)
        tiemp=(n+1.0000*j*Nt)*dt
        tiempo.append(tiemp)
        ptocentr.append(placa[int(N/2),int(N/2)])
        print(tiemp)
        #Actualización del plasmon con la nueva temperatura para la comparativa respecto al original
    
        #Creación de imagenes
        if int(n%frames)==0:
            tiemp=round(n*dt,3)
            
    maximociclo.append(np.max(ptocentr))
    maximociclopos.append(np.argmax(ptocentr))

#Creamos los gifs


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

## CREAMOS LA IMAGEN 3D Y LA GUARDAMOS 
fig2 = plt.figure()
ax2 = plt.axes(projection="3d")
ax2.plot_surface(X, Y,placa, cmap="plasma",edgecolor="none") 
ax2.set_zlim(0,10)
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
    save_all=True,
    append_images=listagif[1:],
    duration=200,         
    loop=0)             
listagif2[0].save(
    "output2.gif",
    save_all=True,
    append_images=listagif2[1:],
    duration=200,
    loop=0)

#Imagen del cambio de temperatura
#vang=round(np.pi*(angula[Nt-1]-angula[0])/Nt/180*1000,4)
plt.figure()
plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Cuatro pasadas")
#plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Ángulo ={ang} mRad/s ")
plt.xlabel("Tiempo (s)")
plt.ylabel("${\Delta}T$")
plt.plot(tiempo,ptocentr)
maximocicloo=[]
for i in range(len(maximociclo)):
    maximocicloo.append(np.round(maximociclo[i],8))
    plt.plot(maximociclopos[i]*dt,maximociclo[i],"rx",label=f"Maximo del ciclo {i+1}= {maximocicloo[i]} $\Delta$T")
#plt.legend()
plt.ylim(0,8) 
plt.savefig("curva mov")
plt.show()

plt.figure()
plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],np.array(plasdef),label="Ajuste según la temperatura")
plt.plot(angulos,Abs,label="Valor inicial de la curva")
plt.title("Cambio de la curva de resonancia ")
plt.legend()
plt.xlim(53,56) #Hay que ajustarlo en función de donde esté el pico para poder apreciar bien la diferencia
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")

plt.show()



print( "TERMINAMOS!!")

for n in range(0,int(Nt/5000)):
            placa=placa.flatten()  
            placa, _ = cgs(A,M@placa) 
            placa=placa.reshape((N,N))
            placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0
            tiemp=((n+(1.0002*j+1)*Nt)*dt)
            tiempo.append(tiemp)
            ptocentr.append(placa[int(N/2),int(N/2)])
            print(tiemp)