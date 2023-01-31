import plotly.express as px
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def primer1():

    LUT = np.sqrt(np.arange(0, 256))
    img = io.imread('lena.jpg')
    fig = px.imshow(img, zmin = 0, zmax = 255, color_continuous_scale='gray')
    fig.show()

    img_sqrt = np.sqrt(img.astype('float'))
    fig = px.imshow(img_sqrt, zmin = 0, zmax = 255, color_continuous_scale='gray')
    fig.show()

#primer1()

def primer2a(gamma):

    img = io.imread('andrewtate.png')
    LUT = np.sqrt(np.arange(0,256))
    c = 255 ** (1 - gamma)
    LUT = c * np.arange(0, 256) ** gamma
    LUT = np.round(LUT).astype('uint8')
    return LUT[img]

def primer2b():

    img = io.imread('andrewtate.png')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    img2 = primer2a(0.5)
    fig = px.imshow(img2, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

#primer2b()

def primer3():

    img = io.imread('gray.jpg')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    LUT = np.arange(256,-1,-1)

    imgn = LUT[img]
    fig = px.imshow(imgn, color_continuous_scale='gray')
    fig.show()

#primer3()

#def primer4():

def zadatak1():

    img = io.imread('stevo.png')
    imga = rescale(img, 1 / 4, order=1, preserve_range=True)
    img2 = resize(imga, img.shape, order=1, preserve_range=True).astype('uint8')
    fig = px.imshow(img2, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    LUT = np.log(np.arange(256)+1)
    LUT = LUT[-1] - LUT
    LUT = LUT/LUT.max() * 255

    img3 = LUT[img2]
    fig = px.imshow(img3,color_continuous_scale='gray')
    fig.show()

#zadatak1()

def zadatak2(p1,p2):

    img = io.imread('tam.jpg').astype('uint8')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    lut1 = np.linspace(0, p1, 5)
    lut2 = np.linspace(p1, p2, 245)
    lut3 = np.linspace(p2, 255, p2-p1)
    LUT = np.concatenate((lut1[:-1], lut2[:-1], lut3), axis=0)

    img = LUT[img]
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

#zadatak2(50,150)


def zadatak3(gamma):

    img = io.imread('tam.jpg')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    LUT = np.sqrt(np.arange(0, 2^16))
    c = 255 ** (1 - gamma)
    LUT = c * np.arange(0, 256) ** gamma
    LUT = np.round(LUT).astype('uint8')



    img = LUT[img]
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

#zadatak3(0.025)

def main():

    print("1. Zad \n2. Zad \n3. Zad\n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        zadatak1()
        main()
    elif n == 2:
        print("*Stavljeno na 50, 150*\n")
        zadatak2(50,150)
        main()
    elif n ==3:
        print("gamma = 0.025 ja tako stavio lol")
        zadatak3(0.025)
        main()

    else:
        print("Greska!")
        main()

main()
