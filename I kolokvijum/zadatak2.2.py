from skimage import io
import plotly.express as px
import numpy as np

img = io.imread('rtg_1.png')
px.imshow(img, color_continuous_scale='gray').show()
min1 = img.min()
max1 = img.max()
opseg = np.arange(min1, max1+1)
gama = 0.5
c = 255**(1-gama)
LUT = c * opseg ** gama
LUT = (LUT - LUT.min()) / (LUT.max() - LUT.min()) * 255

img_gama = LUT[img].astype('uint8')
px.imshow(img_gama, color_continuous_scale='gray').show()
