import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cgs
import matplotlib.pyplot as plt
r = 0.5
N = 500
diagonals = [(1 + 4*r)* np.ones(N*N), -r*np.ones(N*N - 1), -r*np.ones(N*N -1), -r*np.ones(N*N - N), -r*np.ones(N*N - N)]
A = diags(diagonals, [0, -1, 1, N, -N], format="csr")

from skimage.draw import disk

rr, cc = disk((N//2, N//4), N//15)

mask = np.zeros((N, N), dtype = bool)

mask[rr, cc] = 1

rr, cc = disk((N//2, 3*N//4), N//15)

mask[rr, cc] = 1

plt.imshow(mask)



values = np.zeros((N, N))+293

values[mask] = 300


values = values.flatten()

diagonals = [(1-4*r)* np.ones(N*N), r*np.ones(N*N - 1), r*np.ones(N*N -1), r*np.ones(N*N - N), r*np.ones(N*N - N)]


M = diags(diagonals, [0, -1, 1, -N, N], format = "csr")



steps = 1500
arr = np.zeros((steps, N, N))
arr[0] = values.reshape((N, N))

for i in range(1, steps):
    values, _ = (A, M@values)

    arr[i] = values.reshape((N, N))




from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, frameon=False)

X, Y = np.meshgrid(np.arange(0, N), np.arange(0, N))

plot = ax.plot_surface(X, Y, arr[0], cmap="coolwarm")

def updatefig(i, plot):
    ax.clear()
    ax.set_zlim(bottom=0, top = np.max(arr[0]))
    ax.set_axis_off()
    # ax.set_pane_color((0, 0, 0))
    plot = ax.plot_surface(X, Y, arr[i], cmap="coolwarm")
    
    return plot, 
    

ani = FuncAnimation(fig, updatefig, frames=range(0, len(arr), 25), blit=True, fargs=(plot, ))

ani.save('diffusion_gpu.gif', writer = 'ffmpeg', fps = 60)

