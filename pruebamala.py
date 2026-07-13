import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import seaborn as sns
from scipy.sparse.linalg import cgs
from PIL import Image
from io import BytesIO
import sys
from MiCOL import Main_SPR as cr
from Indices import  Index as ri
sys.path.insert(2, 'C:\\Users\Admin\Desktop\Clase (falta pasar el pc antiguo aquí)\Cuarto\TFG\Programas')
Temp=239

ldo,angulo,materiales,grosores = 633.,np.arange(0.,90,1), ['BK7','ag', 'water' ], [ 0., 40. ,0. ] # en nm
a=ri.refindex(ldo,materiales)
r, t, R, T = cr.Multicapa_Matrix(ldo, materiales,Temp, grosores, angulo)
Abs = (1 - R -T.real)
plt.figure()
plt.plot(angulo,Abs)
plt.show()

a=ri.refindex(ldo,materiales)