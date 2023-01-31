from skimage import io, color, exposure
import plotly.express as px
from matplotlib.cm import get_cmap
import numpy as np

img = io.imread("../peppers.png")
px.imshow(img, color_continuous_scale='gray').show()

cmap = get_cmap('jet')
img_color = cmap(img)
img_color = np.uint8(img_color*255)[:,:,:3]
px.imshow(img_color).show()

img_hsv = color.rgb2hsv(img_color)
img_hsv[:,:,1] = exposure.equalize_hist(img_hsv[:,:,1])
px.imshow(img_hsv).show()

img_hsv[:, :, 2] /= 2
px.imshow(img_hsv).show()
img_hsv[:,:,0][(img_hsv[:,:,0] > 0.65) & (img_hsv[:,:,0] < 0.67)] += 15
px.imshow(img_hsv).show()

img_rgb = color.hsv2rgb(img_hsv)
px.imshow(img_rgb).show()
