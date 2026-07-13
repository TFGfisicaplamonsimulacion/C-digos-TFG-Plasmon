import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
####################################
# CAMBIOS FRENTE AL ANTERIOR, EL LASER POR COMODIDAD ESTÁ CENTRADO, AHORA LA FORMA DE INCIDENCIA DEL LASER PUEDE SER UNA ELIPSE
# NO SE SI HACE FALTA HACER ROTACIONES DE ESTA POR COMO ES EL MONTAJE DE LABORATORIO
# ADEMAS INDICA EL TIEMPO SIMULADO, HAY QUE PONER BIEN EL FORMATO Y MIRAR DE AÑADIR PARAMETROS REALES AHORA
# HAY QUE TENER EN CUENTA QUE EL PLASMON DE ALGUNA MANERA ABSORBE CALOR???
#####################################

#parametros
N=50#tamaño que le quiero dar de resolución a la placa
placa=np.zeros((N,N))+293 #la placa por donde la voy a empezar a hacer
pn=np.zeros((N,N)) #Una copia vacia para ir haciendo los calculos
tiempo=150
dt,dx,dy=0.008,10,10
a=dt/dx/dx
b=dt/dy/dy
kx=127
ky=127
t0=293
plt.close('all')

#######################################################################
# crank nicolson mirar ese método
# pillow libreria para que haga un gif con las imagenes
# usar seaborn en lugar de matplotlib, más facil de hacer y que swea bonito
# Mirar
# como alfas son plata y oro (La difusividad térmica)


# cordX=int(input("Indica la coordenada X del centro de donde está el laser: "))
# cordY=int(input("Indica la coordenada Y del centro de donde está el laser: "))
cordX=0
cordY=0

# No pongo que se eligan las coordenadas pq pereza


excx=1
excy=3
x = np.linspace(-N/2, N/2, N)
y = np.linspace(-N/2, N/2, N)
X, Y = np.meshgrid(x, y)
laser = np.exp(-((X-cordX)**2/excx + (Y-cordY)**2/excy)) # Perfil gaussiano del laser

arr = np.zeros((tiempo, N, N))
arr[0] = placa.reshape((N, N))
listagif= []

for n in range(tiempo):
    for i in range(2,N-2):
        for j in range(2,N-2):
            pn[i,j]=placa[i,j]+kx*a*(-placa[i-2,j]/12+4/3*placa[i-1,j]-2.5*placa[i,j]+4/3*placa[i+1,j]-placa[i+2,j]/12)+ky*b*(-placa[i,j-2]/12+4/3*placa[i,j-1]-2.5*placa[i,j]+4/3*placa[i,j+1]-placa[i,j+2]/12)
    # Usar matrices
    
    for i in range(N): #mantener condiciones de contorno
        pn[0,i]=t0
        pn[N-1,i]=t0
        pn[i,0]=t0
        pn[i,N-1]=t0
        pn[1,i]=t0
        pn[N-2,i]=t0
        pn[i,1]=t0
        pn[i,N-2]=t0
    for i in range(N):
        for j in range(0,N):
            pn[i,j] += laser[i,j]*dt*100

    for i in range(0,N):
        for j in range(0,N):
            placa[i,j]=pn[i,j] #copiamos el vector sin hacer un alias
    print(placa)
    arr[i]=placa
    if n%20==0:
        frame=Image.fromarray(placa)
        listagif.append(frame)
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="293.5" ,cmap="plasma")
        plt.show()
        plt.pause(0.001)  
    print("Llevamos", n ," ciclos, es decir:\nLlevamos",n*dt,"segundos")

listagif[0].save(
    "output.gif",
    save_all=True,        # Save multiple frames
    append_images=listagif[1:],  # Add remaining frames
    duration=200,         # Frame duration in milliseconds
    loop=0                # 0 = Loop forever, set to 1 for no loop
)             
from matplotlib.animation import FuncAnimation

arr = np.zeros((tiempo, N, N))
arr[0] = placa.reshape((N, N))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, frameon=False)



plot = ax.plot_surface(X, Y, arr[0], cmap="coolwarm")

def updatefig(i, plot):
    ax.clear()
    ax.set_zlim( top = np.max(arr[0]))
    ax.set_axis_off()
    # ax.set_pane_color((0, 0, 0))
    plot = ax.plot_surface(X, Y, arr[i], cmap="coolwarm")
    
    return plot, 
    

ani = FuncAnimation(fig, updatefig, frames=range(0, len(arr-1), 100), blit=True, fargs=(plot, ))

ani.save('preuba2.gif', writer = 'ffmpeg', fps = 60)