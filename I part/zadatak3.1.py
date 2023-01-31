from skimage import io
import plotly.express as px
import numpy as np
from skimage.transform import rescale

A = io.imread('cookies.png').astype('float')
px.imshow(A, color_continuous_scale='gray').show()

B = A[50:391, 70:261]
B = np.pad(B, 1, 'reflect')
B_f = np.fft.fftshift(np.fft.fft2(B))

laplasijan = np.array([[0, -1, 0], [-1, 5, 1], [0, -1, 0]])
laplasijan = np.pad(laplasijan, ((0, B_f.shape[0]-3), (0, B_f.shape[1]-3)))
laplasijan = np.roll(laplasijan, [-1, -1], axis=(0, 1))
L = np.fft.fftshift(np.fft.fft2(laplasijan))

B_i = B_f * L
B = np.real(np.fft.ifft2(np.fft.ifftshift(B_i)))
B = B[1:-1, 1:-1]

A[50:391, 70:261] = B
A = rescale(A, 2, order=1, preserve_range=True)
px.imshow(A, zmin=0, zmax=255, color_continuous_scale='gray').show()
