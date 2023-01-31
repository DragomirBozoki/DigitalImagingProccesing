import plotly.express as px
from skimage import io
from skimage import color
from skimage import data
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from scipy import signal as sig
import math
from scipy import ndimage
from skimage.transform import rescale, resize

def zadatak1(d):

    img = io.imread('lena.png').astype('uint8')
    img = color.rgb2gray(img)

    N = img.shape[0]
    M = img.shape[1]

    print("===")
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    print("==*==")

    X = N / 2
    Y = M / 2


    img1 = img[round(0):round(X),
                round(0):round(Y)]
    img1 = np.pad(img, pad_width=d, mode='constant', constant_values=0)

    img2 = img[round(X):round(N),
           round(Y):round(M)]

    img2 = np.pad(img, pad_width=d, mode='constant', constant_values=255)

    print("===")

    fig = px.imshow(img1, color_continuous_scale='gray')
    fig.show()
    fig = px.imshow(img2, color_continuous_scale='gray')
    fig.show()






#zadatak1(25)


def imageHistEq(img):
    hist = np.histogram(img, bins=np.arange(257), density=True)[0]
    T = 255 * np.cumsum(hist)
    T = np.round(T).astype('uint8')  # ako zelimo rezultat u tipu uint8

    return T[img], T



def zadatak2():

    from skimage.filters import threshold_otsu


    img = io.imread('eye.jpg')
    fig = px.imshow(img)
    fig.show()

    img = color.rgb2hsv(img)
    fig = px.imshow(img)
    fig.show()

    hist = np.histogram(img, bins=np.arange(257))
    threshhold = threshold_otsu(img)
    binary = img > threshhold

    fig = px.imshow(img[:,:,3])
    fig.show()

zadatak2()