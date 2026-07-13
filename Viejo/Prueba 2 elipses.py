import numpy as np
import matplotlib.pyplot as plt
####################################
# CAMBIOS FRENTE AL ANTERIOR, EL LASER POR COMODIDAD ESTÁ CENTRADO, AHORA LA FORMA DE INCIDENCIA DEL LASER PUEDE SER UNA ELIPSE
# NO SE SI HACE FALTA HACER ROTACIONES DE ESTA POR COMO ES EL MONTAJE DE LABORATORIO
# ADEMAS INDICA EL TIEMPO SIMULADO, HAY QUE PONER BIEN EL FORMATO Y MIRAR DE AÑADIR PARAMETROS REALES AHORA
# HAY QUE TENER EN CUENTA QUE EL PLASMON DE ALGUNA MANERA ABSORBE CALOR???
#####################################

#parametros
tam=100#tamaño que le quiero dar de resolución a la placa
placa=np.zeros((tam,tam))+293 #la placa por donde la voy a empezar a hacer
pn=np.zeros((tam,tam)) #Una copia vacia para ir haciendo los calculos
tiempo=500000
dt,dx,dy=0.002,0.1,0.1
a=dt/dx/dx
b=dt/dy/dy
kx=0.1
ky=0.1
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


excx=int(input("Indica lo alagado que está del laser a mayor valor más alargado en X: \n"))
excy=int(input("Indica la excentricidad del laser \nNegativo elipse en x positivo elipse en y: "))
x = np.linspace(-50, 50, tam)
y = np.linspace(-50, 50, tam)
X, Y = np.meshgrid(x, y)
laser = np.exp(-((X-cordX)**2/excx + (Y-cordY)**2/excy)) # Perfil gaussiano del laser




for n in range(tiempo):
    for i in range(2,tam-2):
        for j in range(2,tam-2):
            pn[i,j]=placa[i,j]+kx*a*(-placa[i-2,j]/12+4/3*placa[i-1,j]-2.5*placa[i,j]+4/3*placa[i+1,j]-placa[i+2,j]/12)+ky*b*(-placa[i,j-2]/12+4/3*placa[i,j-1]-2.5*placa[i,j]+4/3*placa[i,j+1]-placa[i,j+2]/12)
    # Usar matrices
    
    for i in range(tam): #mantener condiciones de contorno
        pn[0,i]=t0
        pn[tam-1,i]=t0
        pn[i,0]=t0
        pn[i,tam-1]=t0
        pn[1,i]=t0
        pn[tam-2,i]=t0
        pn[i,1]=t0
        pn[i,tam-2]=t0
    for i in range(tam):
        for j in range(0,tam):
            pn[i,j] += laser[i,j]*dt

    for i in range(0,tam):
        for j in range(0,tam):
            placa[i,j]=pn[i,j] #copiamos el vector sin hacer un alias
    #print(placa)
    if n%20==0:

        plt.imshow(placa,cmap='plasma')
        plt.colorbar()
        plt.show()
        plt.pause(0.001)  
        print("Llevamos", n ," ciclos, es decir:\nLlevamos",n*dt,"segundos")