from skimage import io
import plotly.express as px
import numpy as np

A = io.imread('deda_mraz.png')
px.imshow(A, color_continuous_scale='gray').show()
N, M = A.shape
B = A[round(0.3*N):round(0.7*M), round(0.3*N):round(0.7*M)]
px.imshow(B, color_continuous_scale='gray').show()

opseg_intenziteta = (155, 255)
broj_binova = 75
korak = 3

img = (B - opseg_intenziteta[0]) / (opseg_intenziteta[1] - opseg_intenziteta[0])
img = img[(img >= 0) & (img <= 1)]
img = np.round(img * (broj_binova-1)).astype('uint8')
img = img[::korak]

histogram = np.zeros(broj_binova)
for vrednost_piksela in img.ravel():
    histogram[vrednost_piksela] += 1

px.line(x=np.arange(broj_binova), y=histogram).show()
