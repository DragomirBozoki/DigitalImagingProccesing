from pandas import array
import plotly.express as px
import numpy as np
import math
from skimage import io
from scipy import ndimage

img = io.imread('slike/lena.png')
fig = px.imshow(img, color_continuous_scale='gray')
fig.show()

rng = np.random.default_rng()

AWGN_noise = rng.standard_normal(size= img.shape)*math.sqrt(100) + 0
img_AWGN = img + AWGN_noise
img_AWGN[img_AWGN > 255] = 255
img_AWGN[img_AWGN < 0] = 0

fig = px.imshow(img_AWGN, color_continuous_scale='gray')
fig.show()

p = rng.uniform(size= img.shape)
p0 = 0.1


img_AWGN[p < p0/2] = 0
img_AWGN[(p0/2<=p) & (p<p0)] = 255

fig = px.imshow(img_AWGN, color_continuous_scale='gray')
fig.show()


def alphaTrimmed(img, w, alpha):

    f = np.zeros(img.shape)
    pad_row = math.floor(w[0]/2)
    pad_col = math.floor(w[1]/2)

    img_p = np.pad(img, ((pad_row, ), (pad_col,)), mode='symmetric')

    for row in range(f.shape[0]):
        for col in range(f.shape[1]):
            region = img_p[row: row + w[0], col: col + w[1]]
            region = np.sort(region, axis=None)

            res = region[alpha//2:-alpha//2].mean()

            f[row, col] = res

    return f


img_final = alphaTrimmed(img_AWGN, (5,5), 15)
fig = px.imshow(img_final, color_continuous_scale='gray')
fig.show()