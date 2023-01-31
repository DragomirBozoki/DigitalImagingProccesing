import numpy as np
from skimage import io
from scipy import ndimage
import plotly.express as px

img = io.imread('cat.png')
sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

G1 = ndimage.convolve(img, sx, mode='nearest', output='float')
G2 = ndimage.convolve(img, sy, mode='nearest', output='float')
G = np.sqrt(G1**2 + G2**2)

gradijent = px.imshow(G, color_continuous_scale='gray')
gradijent.show()
bin = G > 150
binarna = px.imshow(bin, color_continuous_scale='gray')
binarna.show()
