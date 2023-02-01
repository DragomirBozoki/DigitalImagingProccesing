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

def zadatak1():

    img = io.imread('jelka.png')
    img2a = rescale(img, 1/2, order=0, preserve_range=True)
    #img2b = resize(img2a, img, order=0, preserve_range=True)

    fig = px.imshow(img)
    fig.show()

    img4b = rescale(img, 1/4, order=0, preserve_range=True)
    #img2b = resize(img2b, img, order=0, preserve_range=True)

    print("Velicine slika")
    print(img.shape)
    print(img2a.shape)
    print(img4b.shape)

    img2a_p = np.pad(img2a,((0,0), (53,54)))
    img4b_p = np.pad(img2a, ((0,0),(80,80)))

    print("VelicinePADSlika")
    print(img.shape)
    print(img2a_p.shape)
    print(img4b_p.shape)

    fig = px.imshow(img2a_p)
    fig.show()
    fig = px.imshow(img4b_p)
    fig.show()

    img = np.concatenate((img4b_p,img2a_p,img), axis=0)
    fig = px.imshow(img)
    fig.show()

#zadatak1()

def zadatak2(d):

    img = io.imread('paketici1.png').astype('uint8')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    N, M = img.shape

    img2 = img[round(0):round(N * 0.5),
           round(0):round(M * 0.5)]
    img4 = img[round(0.5*N):round(N),
           round(0.5*M):round(M)]

    #fig = px.imshow(img2)
    #fig.show()
    #fig = px.imshow(img4)
    #fig.show()

    img2_p = np.pad(img2,pad_width=d, mode='constant', constant_values=0)
    img4_p = np.pad(img4,pad_width=d, mode='constant', constant_values=255)

    fig = px.imshow(img2_p)
    fig.show()
    fig = px.imshow(img4_p)
    fig.show()

    img2R = resize(img2_p, img2.shape, preserve_range=True)
    img4R = resize(img4_p, img4.shape, preserve_range=True)

    img[round(0):round(N * 0.5),round(0):round(M * 0.5)] = img2R
    img[round(0.5*N):round(N), round(0.5*M):round(M)] = img4R



    fig = px.imshow(img)
    fig.show()

zadatak2(5)