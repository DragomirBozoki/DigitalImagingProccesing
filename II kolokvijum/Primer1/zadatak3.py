from skimage import io
import plotly.express as px
import numpy as np
import math

img = io.imread("../baboon.png")
fig = px.imshow(img, color_continuous_scale='gray')
fig.show()

rng = np.random.default_rng()
p = rng.uniform(size=img.shape)
p0 = 0.1
img1 = img.copy()
img1[p < p0] = 255
fig = px.imshow(img1, color_continuous_scale='gray')
fig.show()


def harmonijskiUsrednjivac(img, kernel):
    i = np.zeros(img.shape)
    pad_rows = math.floor(kernel[0] / 2)
    pad_cols = math.floor(kernel[1] / 2)

    img_p = np.pad(img, ((pad_rows,), (pad_cols,)), mode='symmetric')

    for row in range(i.shape[0]):
        for col in range(i.shape[1]):
            region = img_p[row:row + kernel[0], col:col + kernel[1]]
            i[row, col] = np.prod(kernel) / (1/(region+np.finfo('float').eps)).sum()
    return i


kernel = (3, 5)
img2 = harmonijskiUsrednjivac(img1, kernel)
fig = px.imshow(img2, color_continuous_scale='gray')
fig.show()

psnr = 10*np.log10(255**2/((img - img2)**2).mean())
print("PSNR za harmonijski usrednjivac dimenzije {} je {}".format(kernel, psnr))
