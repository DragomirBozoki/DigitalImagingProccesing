import plotly.express as px
from skimage import io
import numpy as np
from scipy import signal as sig
import math
from scipy import ndimage

def zadatak1a(img):
  kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
  out = np.zeros(img.shape)

  pad_rows = math.floor(3/2)
  pad_cols = math.floor(3/2)

  img_p = np.pad(img, ((pad_rows,),(pad_cols,)), mode="symmetric")

  for row in range(1, out.shape[0]):
    for col in range(1, out.shape[1]):
      region = img_p[row-1:row+2, col-1:col+2]
      #print(region)
      res = (region * kernel).sum()

      out[row-1, col-1] = res

  return out

def zadatak1():
    img = io.imread('lena.png')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    img1=zadatak1a(img)
    fig1 = px.imshow(img1, color_continuous_scale='gray')
    fig1.show()

#zadatak1()

def zadatak2(w,c):

    img = io.imread('lena.jpg')
    kernel = np.ones((w,w))*(1/(w*w))
    Inf = sig.convolve2d(img, kernel, mode='same', boundary='symmetric')

    Ivf = img - Inf

    Ium = img + c * Ivf

    fig = px.imshow(Ium, color_continuous_scale='gray')
    fig.show()

#zadatak2(7,7)

def zadatak3(prag):

    Sy = ([-1,-2,-1],[0,0,0],[1,2,1])
    Sx = ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])
    img = io.imread('lena.jpg').astype('float')

    G1 = ndimage.convolve(img,Sy, mode='reflect')
    fig = px.imshow(G1, color_continuous_scale='gray')
    fig.show()

    G2 = ndimage.convolve(img, Sx, mode='reflect')
    fig = px.imshow(G2, color_continuous_scale='gray')
    fig.show()

    G = np.sqrt(G1**2 + G2**2)
    G_edge =  G > prag

    fig = px.imshow(G_edge, color_continuous_scale='gray')
    fig.show()
#zadatak3(30)


def primer1():

    img = io.imread('tam.jpg')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    img1 = np.pad(img, pad_width=100, mode='constant', constant_values=100)
    fig = px.imshow(img1, color_continuous_scale='gray')
    fig.show()

    img2 = np.pad(img, ((200,), (100,)))
    fig = px.imshow(img2, color_continuous_scale='gray')
    fig.show()

    # 3 - "edge" padding
    img3 = np.pad(img, ((150,), (0,)), mode='edge')
    fig = px.imshow(img3, color_continuous_scale='gray')
    fig.show()

def primer2():

    w = np.ones((101,101))/101**2
    img = io.imread('lena.png')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    img1 = sig.convolve(img, w, mode = 'full')
    fig = px.imshow(img1, color_continuous_scale='gray')
    fig.show()

    w = np.ones((11,11))/11**2
    img2 = sig.convolve2d(img, w, mode='same')
    print(img2.shape)
    fig = px.imshow(img2, color_continuous_scale='gray')
    fig.show()

    w = np.ones((7, 7)) / 7 ** 2
    img3 = sig.convolve2d(img, w, mode='same', boundary='symmetric')
    print(img3.shape)
    fig = px.imshow(img3, color_continuous_scale='gray')
    fig.show()





def main():

    print("1. Zad \n2. Zad \n3. Zad\n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        zadatak1()
        main()
    elif n == 2:
        print("7,7 parametri \n")
        zadatak2(7,7)
        print("bzzzzzzzzz :D")
        main()
    elif n ==3:
        print("25 ja tako stavio lol")
        zadatak3(25)
        main()

    else:
        print("Greska!")
        main()

main()


