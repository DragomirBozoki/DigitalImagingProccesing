import plotly.express as px
from skimage import io
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from scipy import signal as sig
import math
from scipy import ndimage


def zadatak1(g, S_shape):

    f = np.zeros(g.shape)

    pad_rows = math.floor(S_shape[0] / 2)
    pad_cols = math.floor(S_shape[1] / 2)

    img_p = np.pad(g, ((pad_rows,), (pad_cols,)), mode='symmetric')

    for row in range(f.shape[0]):
        for col in range(f.shape[1]):
            region = img_p[row:row + S_shape[0], col:col + S_shape[1]]

            region = np.sort(region, axis=None)

            min = region[0]
            max = region[-1]

            res = (min + max) / 2

            f[row, col] = res

def zadatak1a():
    img = io.imread('lena.png')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()
    # dodavanje Gausa
    rng = np.random.default_rng()
    img_noisy = img + rng.standard_normal(size=img.shape) * np.sqrt(50) + 0
    img_noisy[img_noisy > 255] = 255
    img_noisy[img_noisy < 0] = 0

    S_shape = (5, 5)

    img_filtrirano = zadatak1(img_noisy, S_shape)
    fig_filtrirano = px.imshow(img_filtrirano, zmin=0, zmax=255, color_continuous_scale='gray')
    fig_filtrirano.show()

zadatak1a()
def p1():

    img = np.full((256,256), fill_value=63)
    img[32:32+192,32:32+192] = 127
    img[64:64+128,64:64+128] = 191
    fig = px.imshow(img)
    fig.show()

    hist, bin_edges = np.histogram(img, bins=np.arange(257))
    fig = px.histogram(hist)
    fig.show()

    rng = np.random.default_rng()
    noise = rng.standard_normal(size=img.shape)

    fig = px.imshow(noise, color_continuous_scale='gray')
    fig.show()

    imgnoisy = img + noise*np.sqrt(100)+ 0
    imgnoisy[imgnoisy > 255] = 255
    imgnoisy[imgnoisy < 0] = 0

    karnel1 = np.ones((3,3))/9
    karnel2 = np.ones((5,5))/25

    img_f = ndimage.convolve(imgnoisy.astype('float'), karnel1, mode = 'mirror')

    fig = px.imshow(img_f, color_continuous_scale='gray')
    fig.show()

    return imgnoisy

#zadatak1()

def primer2():

    img = np.full((256,256), fill_value=63)
    img[32:32 + 192, 32:32 + 192] = 127
    img[64:64 + 128, 64:64 + 128] = 191

    hist, bin_edges = np.histogram(img, bins=np.arange(257))
    fig = px.histogram(hist)
    fig.show()

    rng = np.random.default_rng()
    p = rng.uniform(size=img.shape)

    p0 = 0.2

    img_so = img.copy()
    img_so[p<p0] = 255

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=np.stack((img_so, img_so, img_so), axis=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(256), y=hist, mode='lines', name='hist', line=dict(color='blue')), row=1,
                  col=2)
    fig.show()

    psnr = 10 * np.log10(255 ** 2 / ((img - img_so) ** 2).mean())
    print("PSNR za zasumljenu sliku je {}".format(psnr))

primer2()
