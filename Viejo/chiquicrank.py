import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
diago=[8*np.ones(3),3*np.ones(2),4*np.ones(1)]
M=diags(diago,[0,-1,2],format="csc")
A=diags(diago,[0,1,-2],format="csc")
print(M.toarray())
b=np.array([3,2,4])
print(b)
print(A.toarray())
print(M@b)
c=spsolve(A,M@b)
print(c)