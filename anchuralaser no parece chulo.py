import numpy as np
import matplotlib.pyplot as plt
from MiCOL import Main_SPR as cr
Nt=1000
red=20
anc=[]
c=100
for i in range(1000):
    j=np.exp(-((i-633)**2/c))/(np.sqrt(2*np.pi*c/4))
    anc.append(j)

plt.plot(anc)
plt.show()

len(anc)

for i in range(len(anc)):
    ldo,angulo,materiales,grosores = i,np.arange(0.,90,90/Nt/red), ['BK7','ag', 'air' ], [ 0., 40. ,0. ] # en nm
    r, t, R, T = cr.Multicapa_Matrix(ldo, materiales, grosores, angulo)
    