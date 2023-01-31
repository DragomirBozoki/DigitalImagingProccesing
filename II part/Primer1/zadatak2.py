from skimage import io
import plotly.express as px
import numpy as np
import math


def localTresh(img, kernel):
    T = np.zeros(img.shape)
    pad_rows = math.floor(kernel[0] / 2)
    pad_cols = math.floor(kernel[1] / 2)

    img_p = np.pad(img, ((pad_rows,), (pad_cols,)), mode='symmetric')

    for row in range(T.shape[0]):
        for col in range(T.shape[1]):
            region = img_p[row:row + kernel[0], col:col + kernel[1]]
            m = np.mean(region)
            s = np.std(region)
            T[row, col] = m + 0.25*s + 4
    return T


img = io.imread("../cellphone.png")
px.imshow(img, color_continuous_scale='gray').show()
kernel = (5, 5)
T = localTresh(img, kernel)
res = img > T
px.imshow(res, color_continuous_scale='gray').show()
