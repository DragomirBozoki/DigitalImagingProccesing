import plotly.express as px
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def zadatak1a(bn, go, k):

    img = io.imread('tam.jpg')
    img = (img - go[0]) / (go[1]-go[0])
    img = img[(img >= 0)&(img <1)]

    img = img[::k]
    img = np.round(img*(bn-1)).astype('uint8')

    h = np.zeros(bn)
    for i in range(img.size):
        h[img[i]] = h[img[i]] + 1

    return h

zadatak1a(2, [100, 200], 5)

def zadatak1():

    img = io.imread('tam.jpg')
    h = zadatak1a(50, [100, 200], 5)

    fig = px.line(x = np.linspace(100, 200, 50), y = h)
    fig.show()


def primer1():

    img = io.imread('tam.jpg')

    counter4 = (img == 50).sum()
    print(counter4)

    print('There are {} with intensity of 50 in image'.format(counter4))

#primer1()

def primer2():

    img = io.imread('tam.jpg').astype('uint8')
    fig = px.imshow(img,color_continuous_scale='gray')
    fig.show()

    h = np.zeros(256)
    for pixel_intensity in img.ravel():
        h[pixel_intensity] += 1

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(256), y=h, mode='lines'), row=1, col=1)
    fig.show()

#primer2()

def primer3():

    img = io.imread('tam.jpg').astype('uint8')
    fig = px.imshow(img,color_continuous_scale='gray')
    fig.show()

    h = np.zeros(256)
    for pixel_intensity in img.ravel():
        h[pixel_intensity] += 1

    nh = h/h.sum()

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=np.arange(256), y=h, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(256), y=nh, mode='lines'), row=1, col=2)

    fig.show()

#primer3()

def primer4():

    img = io.imread('tam.jpg')

    counter4 = (img <= 50).sum()
    print(counter4)

    print('There are {} with intensity of 50 or lower in img'.format(counter4))

    hist, bin_edges = np.histogram(img, bins=np.arange(257), density=True)
    fig = px.line(x=np.arange(256), y=hist)
    fig.show()

#primer4()

def main():

    print("1. Zad \n2. Zad \n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        zadatak1()
        main()
    if n == 2:
        print("Neko ima ovo uradjen a taj covek nisam ja </3")
   # elif n == 2:
       # print("*Stavljeno na 50, 150*\n")
       # zadatak2(50,150)
      #  main()

    else:
        print("Greska!")
        main()

main()