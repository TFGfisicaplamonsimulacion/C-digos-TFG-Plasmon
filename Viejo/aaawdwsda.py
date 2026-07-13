import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inv


'''
Método ADI para la ecuación de difusión en 2D
'''

N = 20; dimt = 1000
dt = 0.01; dx = 0.9; dy = 0.9
alfax = 0.5; alfay = 0.5
sx = alfax * dt/dx**2/2; sy = alfay * dt/dy**2/2

X = np.zeros([N+1, N+1])
Y = np.zeros([N+1,N+1])
T = np.zeros([N+1, N+1])
Ti = np.zeros([N+1,N+1])
Tn = np.zeros([N+1,N+1])
T[9:11, 9:11] = 12

plt.close('all')
plt.figure(1)
plt.title('Ecuación de difusión en 2D. Método ADI.')
plt.xlabel('indice')
plt.ylabel('Temp')
plt.grid(True)
plt.minorticks_on()

for i in range(N+1):
        if i == 0 or i == N:
            X[i,i] = 1
            Y[i,i] = 1
        else:
            X[i,i-1] = -sx
            X[i,i] = 1 + 2*sx
            X[i,i+1] = -sx
            Y[i,i-1] = -sy
            Y[i,i] = 1 + 2*sy
            Y[i,i+1] = -sy

for n in range(dimt):
    for i in range(1,N):
        for j in range(1,N):
            Ti[i,j] = sy * T[i,j-1] + (1 - 2*sy) * T[i,j] + sy * T[i,j+1]
    
    Tn = np.dot(inv(X),Ti)
    T = np.copy(Tn)
    for i in range(1,N):
        for j in range(1,N):
            Ti[i,j] = sy * T[i-1,j] + (1 - 2*sx) * T[i,j] + sx * T[i+1,j]
            
    Tn = np.dot(inv(Y), Ti)
    T[0,:] = T[1,:]
    T[:,0] = T[:,1]
    T[N,:] = T[N-1,:]
    T[:,N] = T[:,N-1]
    T = np.copy(Tn)
    if n % 10 == 0:
        plt.figure(1)
        plt.pcolor(T)
    plt.show
            