import math

import plotly.express as px
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

#STISNI RUN SAM CE ODRADITI OSTALO
#


def zadatak1(x,y):

    img = io.imread('tam.jpg')
    fig = px.imshow(img, color_continuous_scale='gray')



    N = img.shape[0]
    M = img.shape[1]

    img2 = img.copy()
    img2 = img2[round(N * 101):round(N * 101),
           round(M * 101):round(M * 101)]

    fig = px.imshow(img2, color_continuous_scale='gray')
    fig.show()



    img_2 = rescale(img2, 4, order=1, preserve_range=True)
    img_2b = resize(img_2, img.shape, order=1, preserve_range=True)

    fig = px.imshow(img_2b)
#zadatak1(50,50)

def zadatak2(b):
    print("bzzzzzzzz :D")

    img = io.imread('doomer.jpg')

    img_2a = rescale(img, 1 / 5, order=0, preserve_range=True)
    img_2b = resize(img_2a, img.shape, order=0, preserve_range=True)

    fig = px.imshow(img_2b, color_continuous_scale='gray')
    fig.show()

    bit_num = b  # broj bita
    bin_string = '1' * bit_num + '0' * (8-bit_num)

    quant_mask = int(bin_string, 2)

    Q = img & quant_mask
    Q = np.round(Q / quant_mask * 255).astype('uint8')

    fig = px.imshow(Q, color_continuous_scale='gray')
    fig.show()


    img_2b = resize(Q, img.shape, order=0, preserve_range=True)

    fig = px.imshow(Q, color_continuous_scale='gray')
    fig.show()

    print('Dimenzija slike img: {}'.format(img_2a.shape))
    print('Dimenzija slike Q: {}'.format(Q.shape))
    print('Dimenzija slike img_2b: {}'.format(img_2b.shape))

#zadatak2(2)

def zadatak3():

    xpr = np.arange(0,51,1)
    ypr = np.arange(0,51,1)

    x, y = np.meshgrid(xpr, ypr)

    #cityblock
    De = np.abs(x-25) + np.abs(y-25)
    fig = px.imshow(De, color_continuous_scale='gray')
    fig.show()

    #Chessboard
   # De= np.abs(x-25) + np.abs(y-25)


   # DeC = np.max(De)



  #  fig = px.imshow(DeC,color_continuous_scale='gray')
  #  fig.show()

#zadatak3()

def primer1():

    img = io.imread('doomer.jpg')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    bit_num = 3
    bin_string = '1' * bit_num + '0'*(8-bit_num)

    quant_mask = int(bin_string, 2)
    print('Decimalni broj za kvantizacionu masku: {}'.format(quant_mask))

    Q = img & quant_mask
    Q = np.round(Q/quant_mask*255).astype('uint8')

    print('Unikatne vrednosti u okviru kvantizovane slike: {}'.format(np.unique(Q)))

    fig = px.imshow(Q, color_continuous_scale='gray')
    fig.show()

#zadatak1()

def primer2():

    img =io.imread('chad.jpg')
    fig = px.imshow(img, color_continuous_scale='gray')

    fig.show()

    img_2 = rescale(img, 1 / 4, order=1, preserve_range=True)
    img_2b = resize(img_2, img.shape, order=1, preserve_range=True)
    img_3 = rescale(img, 1/8, order=1, preserve_range=True)
    img_3b = resize(img_3, img.shape, order=1, preserve_range=True)
    img_4 = rescale(img, 1/16, order=1, preserve_range=True)
    img_4b = resize(img_4, img.shape, order=1, preserve_range=True)

    img_2b = np.round(img_2b).astype('uint8')
    img_3b = np.round(img_3b).astype('uint8')
    img_4b = np.round(img_4b).astype('uint8')

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_2b)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_3b)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(img_4b)


#primer2()

def primer4():

    x_prom = np.arange(0,41,1)
    y_prom = np.arange(0,41,1)

    x, y = np.meshgrid(x_prom,y_prom)

    print(x)
    print("---")
    print(y)

    De = np.sqrt((x - 20) ** 2 + (y - 20) ** 2)

    fig = px.imshow(De, color_continuous_scale='gray')
    fig.show()

#primer4()

def primer5():

    img = io.imread('lena.png')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    y = np.arange(img.shape[0]).reshape(-1, 1) - np.floor(img.shape[0] / 2)
    x = np.arange(img.shape[1]) - np.floor(img.shape[1] / 2)

    De = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)
    M = De < np.floor(img.shape[0] / 2)

    fig = px.imshow(M, color_continuous_scale='gray')
    fig.show()

    img_masked1 = img * M
    fig = px.imshow(img_masked1, color_continuous_scale='gray')
    fig.show()

    img_masked2 = img.copy()
    img_masked2[~M] = 0
    fig = px.imshow(img_masked2, color_continuous_scale='gray')
    fig.show()

#primer5()


def main():

    print("1. Zad \n2. Zad \n3. Zad\n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        print("Uopste mi nije jasno sta on tacno trazi od mene\nAko skontam cu ispravim ovaj zadatak :]")
        print("Do tada evo malo retroWave-a: https://www.youtube.com/watch?v=QHNakk1oM7g\n*\n")
        main()
    elif n == 2:
        print("*Stavljeno na 3bita*\n")
        zadatak2(3)
        main()
    elif n ==3:
        zadatak3()
        main()

    else:
        print("Greska!")
        main()

main()
