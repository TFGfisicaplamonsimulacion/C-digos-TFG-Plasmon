import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
from MiCOL import Main_SPR_no_va as cr



#Forma de la placa y parametros
L,N,alpha,dt,T_final=100,100,127,0.05,30 #Longitud, numero de casillas alpha del materia, tiempos
dx, dy = L/(N-1), L/(N-1)  # Separaciones
Nt,r1 = int(T_final/dt),alpha*dt/2/dx/dy  # Numero de pasos
placa=np.ones((N,N))-1 #la placa por donde la voy a empezar a hacer


tempera=27 #Temperatura ambiente en celsius


red=30 #factor por el que reducimos el ángulo por el que barremos 30 sería 1.5 grados en cada dirección

# parametros del plasmon
ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/Nt/red) ,['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm ldo y grosores
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo, tempera=tempera,mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real) #Curva de absorcion




maxi=np.argmin(R)
maxi=np.argmin(abs(angulos-49))
cl=maxi*90/len(Abs)  #centro del angulo en torno al que movemos el laser

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


anch=9.9 #usamos un filtro para tener solo un ancho espectral de 10 nm por ejemplo
ptores=[]
tiempo=[]
ptocentr=[]
plasdef=[]
angula=[]
listagif=[]
listagif2=[]
frames=int(Nt/10)
teq=[]
abseq=[]
for k in range(40,60):
    maxi=np.argmin(abs(angulos-k))
    cl=maxi*90/len(Abs)
    abseq.append(Abs[maxi])
    for n in range(0,Nt):
        ang=round((n*90/Nt-45)/red+cl,3)
        ang=(maxi)*90/len(Abs)

        #Laser ajustandose en cada paso
        excx=(N/10*0.4246*np.cos(ang*2*np.pi/360))**2
        excy=(N/10*0.4246)**2
        laser = np.exp(-((X-cordX)**2/(2*excx) + (Y-cordY)**2/(2*excy)))# Perfil gaussiano 
        laser=laser/np.sum(laser)
    
        #para anchos espectreales sum((Abs)[int(n+len(Abs)*(cl-45/red)/90),:])
        #Operaciones 
        #placa=placa+laser*(Abs)[int(n+len(Abs)*(cl-45/red)/90)]*0.06*anch*dt/2.492/(1e-4)#LASER EN MOVIMIENTO
        placa=placa+laser*(Abs)[maxi]*0.06*anch*dt/2.492/(np.pi*1e-4) #LASER QUIETO EN UN PUNTO
        placa=placa.flatten()  
        placa, _ = cgs(A,M@placa) 
        placa=placa.reshape((N,N))
        placa[:,N-1],placa[N-1,:],placa[:,0],placa[0,:]=0,0,0,0 #Condiciones de contorno
        
        
        angula.append(ang)
        tiemp=n*dt
        tiempo.append(tiemp)
        ptocentr.append(placa[int(N/2),int(N/2)])
    
        #Actualización del plasmon con la nueva temperatura para la comparativa respecto al original
        #tempmed=[]
        #for i in range(int(N/2-9),int(N/2+10)):
        #    for j in range(int(N/2-9),int(N/2+10)):
        #        tempmed.append(placa[i,j])  
        #tempera=np.mean(tempmed)+27    
        #r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
        #Abs = (1 - R -T.real)
        #ptores.append(np.argmin(R)*90/len(Abs))
        #Absdef= (1 - R[int(n+len(Abs)*(cl-45/red)/90)] -T[int(n+len(Abs)*(cl-45/red)/90)].real)
        #plasdef.append(Absdef)
        print(n)
    
    
    
    
    
        #Creación de imagenes
        if int(n%frames)==0:
            tiemp=round(n*dt,3)

    teq.append(ptocentr[-1])
    print(f"Ángulo {k}")
#Creamos los gifs


plt.plot(teq,abseq
         )
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
vang=round(np.pi*(angula[Nt-1]-angula[0])/Nt/180*1000,4)
plt.figure()
plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Velocidad angular ={vang} mRad/s ")
#plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Ángulo ={ang} mRad/s ")
plt.xlabel("Tiempo (s)")
plt.ylabel("${\Delta}T$")
plt.plot(tiempo,ptocentr)

plt.ylim(0,3) 
plt.show()


print( "TERMINAMOS!!")

plt.figure()
#plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],np.array(plasdef),label="Ajuste según la temperatura")
plt.plot(angulos,Abs,label="Valor inicial de la curva")
plt.title("Cambio de la curva de resonancia ")
plt.legend()
plt.xlim(52,56)
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("cambio abs")
plt.show()

plt.figure()
plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],(Abs[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))]-np.array(plasdef)))
plt.legend()
plt.xlim(52,56)
plt.xlabel("Ángulo θ")
plt.ylabel("Diferencia en la absorción ")

plt.show()

plt.figure()
plt.plot(tiempo,ptores)
plt.xlabel("Tiempo")
plt.ylabel("Ángulo de resonancia θ")

plt.show()