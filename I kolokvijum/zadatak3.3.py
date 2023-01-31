import numpy as np
from skimage import io
import plotly.express as px

img = io.imread('lena.png')

histogram = np.zeros(256)
for vrednos_piksela in img.ravel():
    histogram[vrednos_piksela] += 1

hist = px.line(x=np.arange(256), y=histogram)
hist.show()
