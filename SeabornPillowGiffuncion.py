import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
####################################
# HAY QUE TENER EN CUENTA QUE EL PLASMON DE ALGUNA MANERA ABSORBE CALOR???
#####################################

#parametros
N=50#tamaño que le quiero dar de resolución a la placa
placa=np.zeros((N,N))+293 #la placa por donde la voy a empezar a hacer
pn=np.zeros((N,N)) #Una copia vacia para ir haciendo los calculos
tiempo=1500
dt,dx,dy=0.08,10,10
a=dt/dx/dx
b=dt/dy/dy
kx=127
ky=127
t0=293
plt.close('all')

#######################################################################
# crank nicolson mirar ese método


# cordX=int(input("Indica la coordenada X del centro de donde está el laser: "))
# cordY=int(input("Indica la coordenada Y del centro de donde está el laser: "))
cordX=0
cordY=0


excx=1
excy=30
x = np.linspace(-N/2, N/2, N)
y = np.linspace(-N/2, N/2, N)
X, Y = np.meshgrid(x, y)
laser = np.exp(-((X-cordX)**2/excx + (Y-cordY)**2/excy)) # Perfil gaussiano del laser


listagif= []
listagif2 =[]
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
            pn[i,j] += laser[i,j]*dt

    for i in range(0,N):
        for j in range(0,N):
            placa[i,j]=pn[i,j] #copiamos el vector sin hacer un alias

    
    
    if n%20==0:
        
        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="293",vmax="310" ,cmap="plasma",xticklabels=5,yticklabels=5)
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
        ax2.set_zlim(np.min(placa), 310)
        plt.gca().invert_yaxis()

        

        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()  # Free memory

        img2 = Image.open(buf2)
        listagif2.append(img2)
        print("Llevamos", n ," ciclos, es decir:\nLlevamos",n*dt,"segundos")
        
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
