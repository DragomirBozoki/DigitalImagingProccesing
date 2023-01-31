import plotly.express as px
from skimage import io
import numpy as np

def primer1():

    img = io.imread('lena.png')
    Gnew = img[:,:,1].astype('float') + 200
    Gnew[Gnew>255] = 255

    img1 = img.copy()
    img1[:,:,1] = Gnew
    fig = px.imshow(img1)
    fig.show()
#primer1()

def primer2():

    img = io.imread('baboon.png').astype('uint8')
    fig = px.imshow(img)
    fig.show()

    r = np.abs(img[:,:,0].astype('float') - 230) <= 30
  #  fig = px.imshow(r)
  #  fig.show()

    g = np.abs(img[:,:,1].astype('float') - 70) <= 30
    b = np.abs(img[:, :, 2].astype('float') - 50) <= 30

    m = r & g & b
    fig = px.imshow(m)
    fig.show()

    imgm = img * np.stack((m,m,m), axis = 2)
    fig = px.imshow(imgm)
    fig.show()

#primer2()

def primer4(img):


    img = img / 255
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    H = np.rad2deg(
        np.arccos(0.5 * (2 * R - G - B) / (np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + np.finfo('float').eps)))

    H[B>G] = 360 - H[B>G]

    Cmin = img.min(axis=2)
    GrayMask = (R == G) & (G == B)
    H[GrayMask] = 0
    S = 1 - 3 / (R + G + B + np.finfo('float').eps) * Cmin
    S[GrayMask] = 0
    I = (R+G+B) / 3

    return np.stack((H,S,I), axis=2)

def primer3a():

    img = io.imread('baboon.png')
    img = primer4(img)
    fig = px.imshow(img)
    fig.show()

primer3a()