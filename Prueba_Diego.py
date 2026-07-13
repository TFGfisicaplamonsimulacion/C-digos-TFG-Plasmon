
import numpy as np
# import pandas as pd
# import time
import sys

sys.path.insert(1, 'C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela\\Instrumentación_Laboratorio\\Simulacion')
sys.path.insert(2, 'C:\\Users\\Usuario\\OneDrive - Universidade de Santiago de Compostela (1)\\Instrumentación_Laboratorio\\Simulacion')

from MiCOL import Main_SPR as cr

# import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Parametros iniciales
# =============================================================================
ldo,angulo,materiales,grosores = np.arange(623.,634.,1),np.arange(0.,90,0.1) ,['BK7','au', 'air' ], [ 0., 40. ,0. ] 



r, t, R, T = cr.Multicapa_Matrix(ldo, materiales, grosores, angulo)
A = 1 - R -T
plt.figure()
plt.plot(angulo,A)

plt.show()