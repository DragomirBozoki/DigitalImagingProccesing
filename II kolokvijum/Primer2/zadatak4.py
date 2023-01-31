from skimage import io, color, exposure
from skimage.util import img_as_ubyte
import plotly.express as px
from matplotlib.cm import get_cmap

img = io.imread("../peppers.png")
px.imshow(img, color_continuous_scale='gray').show()

cmap = get_cmap('jet', lut=256)
img_color = cmap(img)
px.imshow(img_color).show()

img_hsv = color.rgb2hsv(img_color)
img_eq = img_as_ubyte(exposure.equalize_hist(img_hsv[:, :, 1]))
px.imshow(img_hsv).show()

img_hsv[:, :, 2] /= 2
px.imshow(img_hsv).show()

img_rgb = color.hsv2rgb(img_hsv)
img_rgb[:, :, 2] += 15
px.imshow(img_rgb).show()
