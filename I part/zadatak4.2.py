from skimage import io
import numpy as np
from scipy import ndimage
import plotly.express as px

A = io.imread('knee1.png')
A = A.astype('float')
px.imshow(A, color_continuous_scale='gray').show()

arit_usr = np.ones((5, 5)) / 5**2
B = ndimage.convolve(A, arit_usr, mode='mirror')

sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

G1 = ndimage.convolve(A.astype('float'), sy, mode='mirror')
G2 = ndimage.convolve(A.astype('float'), sx, mode='mirror')
G = np.sqrt(G1**2 + G2**2)

I = G > 90
px.imshow(I, color_continuous_scale='gray').show()

C = np.zeros(A.shape)
for y in range(A.shape[0]):
    for x in range(A.shape[1]):
        if I[y, x]:
            C[y, x] = A[y, x]
        else:
            C[y, x] = B[y, x]
px.imshow(C, zmin=0, zmax=255, color_continuous_scale='gray').show()

