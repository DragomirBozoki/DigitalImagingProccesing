import plotly.express as px
from skimage import io
import numpy as np
import math

img = io.imread('../lena.png')

# Gausov sum
rng = np.random.default_rng()
img_noisy = img + rng.standard_normal(size=img.shape)*np.sqrt(100)
img_noisy[img_noisy > 255] = 255
img_noisy[img_noisy < 0] = 0

# So i biber sum
p = rng.uniform(size=img.shape)
p0 = 0.1
img_noisy[p < p0/2] = 0
img_noisy[(p0/2 <= p) & (p < p0)] = 255

fig = px.imshow(img_noisy, color_continuous_scale='gray')
fig.show()

kernel = (5, 5)
f = np.zeros(img.shape)
alpha = 4 #ne pravi razliku koliko alfa se stavi

pad_rows = math.floor(kernel[0]/2)
pad_cols = math.floor(kernel[1]/2)

img_p = np.pad(img, ((pad_rows,), (pad_cols,)), mode='symmetric')

for row in range(f.shape[0]):
    for col in range(f.shape[1]):
        region = img_p[row:row+kernel[0], col:col+kernel[1]]
        region = np.sort(region, axis=None)
        res = region[alpha//2:-alpha//2].mean()
        f[row, col] = res

fig = px.imshow(f, color_continuous_scale='gray')
fig.show()
