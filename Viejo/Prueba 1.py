import numpy as np
import matplotlib.pyplot as plt


#####################################
#parametros
tam=100 #tamaño que le quiero dar
placa=np.zeros((tam,tam))+293 #la placa por donde la voy a empezar a hacer
pn=np.zeros((tam,tam)) #Una copia vacia para ir haciendo los calculos
tiempo=500000
dt,dx,dy=0.0002,0.1,0.1
a=dt/dx/dx
b=dt/dy/dy
kx=0.1
ky=0.1
t0=293
plt.close('all')
#######################################################################


cordX=int(input("Indica la coordenada X del centro de donde está el laser: "))
cordY=int(input("Indica la coordenada Y del centro de donde está el laser: "))

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
laser = np.exp(-(X**2 + Y**2))*dt*0.1  # Perfil gaussiano del laser

# Calcular las esquinas de la región donde se colocará el láser
inicio_x = cordX - laser.shape[0] // 2
fin_x = inicio_x + laser.shape[0]
inicio_y = cordY - laser.shape[1] // 2
fin_y = inicio_y + laser.shape[1]

#problema no es circular del todo el laser, la region en la que lo aplico es cuadrada xD


for n in range(tiempo):
    for i in range(2,tam-2):
        for j in range(2,tam-2):
            pn[i,j]=placa[i,j]+kx*a*(-placa[i-2,j]/12+4/3*placa[i-1,j]-2.5*placa[i,j]+4/3*placa[i+1,j]-placa[i+2,j]/12)+ky*b*(-placa[i,j-2]/12+4/3*placa[i,j-1]-2.5*placa[i,j]+4/3*placa[i,j+1]-placa[i,j+2]/12)
    
    
    for i in range(tam): #mantener condiciones de contorno
        pn[0,i]=t0
        pn[tam-1,i]=t0
        pn[i,0]=t0
        pn[i,tam-1]=t0
        pn[1,i]=t0
        pn[tam-2,i]=t0
        pn[i,1]=t0
        pn[i,tam-2]=t0
        pn[inicio_x:fin_x, inicio_y:fin_y] += laser

    for i in range(0,tam):
        for j in range(0,tam):
            placa[i,j]=pn[i,j] #copiamos el vector sin hacer un alias
    #print(placa)
    if n%20==0:

        plt.imshow(placa,cmap='plasma')
        plt.colorbar()
        plt.show()
        plt.pause(0.001)  
        print("Llevamos", n ," ciclos")