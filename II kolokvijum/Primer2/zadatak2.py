import plotly.express as px
from skimage import io, morphology
import numpy as np
from scipy import ndimage

img = io.imread('../lena.png')

R1 = np.array([[-1, 0], [0, -1]])
R2 = np.array([[0, -1], [-1, 0]])

G1 = ndimage.convolve(img, R1, mode='mirror')
fig = px.imshow(G1, color_continuous_scale='gray')
fig.show()

G2 = ndimage.convolve(img, R2, mode='mirror')
fig = px.imshow(G2, color_continuous_scale='gray')
fig.show()

G = np.abs(G1) + np.abs(G2)
fig = px.imshow(G, color_continuous_scale='gray')
fig.show()

T = 2*np.sqrt((G**2).mean())
G_edge = G > T
fig = px.imshow(G_edge, color_continuous_scale='gray')
fig.show()

G_thin = morphology.thin(G_edge)
fig = px.imshow(G_thin, color_continuous_scale='gray')
fig.show()
