import plotly.express as px
from skimage import io, color, morphology
import numpy as np

img = io.imread('../elementi.bmp')
fig = px.imshow(img)
fig.show()

img_hsv = color.rgb2hsv(img)
# #H kanal
# fig = px.imshow(img_hsv[:, :, 0], color_continuous_scale='gray')
# fig.show()
# #S kanal
# fig = px.imshow(img_hsv[:, :, 1], color_continuous_scale='gray')
# fig.show()
# #V kanal
# fig = px.imshow(img_hsv[:, :, 2], color_continuous_scale='gray')
# fig.show()

mb = (img_hsv[:, :, 0] > 0.25) & (img_hsv[:, :, 0] < 0.36) & (img_hsv[:, :, 1] > 0.1)
fig = px.imshow(mb, color_continuous_scale='gray')
fig.show()

se1 = morphology.disk(1)
img_o = morphology.opening(mb, se1)
se2 = morphology.disk(7)
img_c = morphology.closing(mb, se2)
fig = px.imshow(img_c, color_continuous_scale='gray')
fig.show()

crvena = img[:, :, 0]
zelena = img[:, :, 1]
plava = img[:, :, 2]

c = crvena[img_c].mean()
z = zelena[img_c].mean()
p = plava[img_c].mean()

img_fig = np.zeros(img.shape)
img_fig[:, :, 0][img_c] = c
img_fig[:, :, 1][img_c] = z
img_fig[:, :, 2][img_c] = p
fig = px.imshow(img_fig) #kursor mi ne pokazuje vrednosti piksela
fig.show()
