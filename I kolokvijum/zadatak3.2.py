from skimage import io
import plotly.express as px
import numpy as np

img = io.imread('hand.png')
px.imshow(img, color_continuous_scale='gray').show()

histogram = np.histogram(img, bins=np.arange(257), density=True)[0]
prag = 255
suma = 0
for i in range(255, -1, -1):
    suma += histogram[i]
    if suma > 0.25:
        prag = i
        break

img_bin = img > prag
px.imshow(img_bin, color_continuous_scale='gray').show()
