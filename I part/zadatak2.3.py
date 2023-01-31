from skimage import io
import numpy as np
import plotly.express as px

img = io.imread('cat.png')
lut1 = np.linspace(0, 30, 71)
lut2 = np.linspace(30, 200, 200-70+1)
lut3 = np.linspace(200, 255, 256-200)
LUT = np.concatenate((lut1[:-1], lut2[:-1], lut3), axis=0)
LUT_n = 255 - LUT

img_n = LUT_n[img]
fig = px.imshow(img_n, color_continuous_scale='gray')
fig.show()
