from skimage import io, color
import plotly.express as px
import numpy as np
import math

img = io.imread("../simboli.png")
px.imshow(img).show()

img_hsv = color.rgb2hsv(img)
# h = img_hsv[:, :, 0]
# px.imshow(h, color_continuous_scale='gray').show()

img_hsv[:, :, 0][img_hsv[:, :, 0] == 2/3] = 1/6
px.imshow(img_hsv[:, :, 1], color_continuous_scale='gray').show()

s = img_hsv[:, :, 1]
kernel = (86, 86)
pad_rows = math.floor(kernel[0] / 2)
pad_cols = math.floor(kernel[1] / 2)

img_p = np.pad(s, ((pad_rows,), (pad_cols,)))

for row in range(s.shape[0]):
    for col in range(s.shape[1]):
        region = img_p[row:row + kernel[0], col:col + kernel[1]]
        if region.sum() == np.prod(region.shape):
            img_hsv[:, :, 0][(row - pad_rows):(row + pad_rows), (col - pad_cols):(col + pad_cols)] = 2/3

img_rgb = color.hsv2rgb(img_hsv)
px.imshow(img_rgb).show()
