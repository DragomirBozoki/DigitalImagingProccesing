import plotly.express as px
from skimage import io
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from scipy import signal as sig
import math
from scipy import ndimage
from skimage.transform import rescale,resize

def zadatak1A(img):

    img = io.imread(img)

    img_norm = (img - img.min())/(img.max()-img.min())
    return img_norm

#zadatak1A()

def zadatak1B(mini,maxi):

    img = zadatak1A('tam.jpg')
    fig = px.imshow(img)
    fig.show()

    imgrange = (img - mini)/(maxi - mini)  * maxi
    fig = px.imshow(imgrange)
    fig.show()


#zadatak1B(0,255)


def zadatak2(x,y):


    img = io.imread('tam.jpg')
    fig = px.imshow(img)
    fig.show()

    img1 = img[x-50:x+50,
    y-50:y+50]

    img1 = rescale(img1, 4/1, order=0, preserve_range=True)
    img1a= resize(img1, img.shape, order=0, preserve_range=True)
    fig = px.imshow(img1a)
    fig.show()

#zadatak2(100,100)

def zadatak3(p1,p2):

    img1a = io.imread('tam.jpg')
    fig = px.imshow(img1a)
    fig.show()

    lut1 = np.linspace(0,p1,6)
    lut2 = np.linspace(p1, p2, 246)
    lut3 = np.linspace(p2, 257, 257)

    LUT = np.concatenate((lut1[:-1],lut2[:-1],lut3),axis=0)
    img2 = LUT[img1a]
    fig = px.imshow(img2)
    fig.show()

#zadatak3(50,175)

def zadatak4():

    img = io.imread('lena.jpg').astype('uint8')
    hist = np.histogram(img, bins=np.arange(257), density=True)[0]
    mini = 0.02
    maxi = 0.003

    h = np.zeros(256)
    for pixel_intensity in img.ravel():
        h[pixel_intensity] += 1

    h_norm = h/h.sum()
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=np.arange(256), y=h, mode='lines', name='histogram'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(256), y=h_norm, mode='lines', name='norm hist'), row=1, col=2)
    fig.show()

    #12
    #225

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] < 25:
                img[y, x] = 0
            elif img[y, x] > 225:
                img[y, x] = 255

    fig = px.imshow(img)
    fig.show()

#zadatak4()

def zadatak5():

    img = io.imread('lena.jpg').astype('float')
    fig = px.imshow(img)
   # fig.show()

    Sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    print(Sy)
    print(Sx)

    GX = ndimage.convolve(img,Sy, mode='reflect')
    GY = ndimage.convolve(img, Sx, mode='reflect')

    G = GX * GY

    T = 10 #prag
    G_edge = G > T
    fig = px.imshow(G_edge, color_continuous_scale='gray')
    fig.show()

#zadatak5()

def zadatak6(D0, C):

    img = io.imread('lena.jpg')
    H_size = img.shape
    if C < 0:
        print("Greska, C mora biti vece/jednako 0")
    else:
        u = np.arange(H_size[0]).reshape(-1, 1) - np.floor(H_size[0] / 2)
        v = np.arange(H_size[1]) - np.floor(H_size[1] / 2)

        D = np.sqrt(u ** 2 + v ** 2)
        Hvf = np.exp(-D ** 2 / (2 * D0 ** 2))

        Hum = 1 + C * Hvf

        IMG = np.fft.fftshift(np.fft.fft2(img))
        IMG1 = Hum * IMG

        img1 = np.fft.ifft2(np.fft.fftshift(IMG1))
        fig = px.imshow(np.log(np.abs(img1) + 1), color_continuous_scale='gray')
        fig.show()

#zadatak6(5,3)

def zadatak6_4():

    img = io.imread('lena.jpg')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    IMG = np.fft.fftshift(np.fft.fft2(img))
    pad = 1

    Sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    Sy = np.pad(Sy, ((0, IMG.shape[0] - 3), (0, IMG.shape[1] - 3)))
    Sy = np.roll(Sy, [-pad, -pad], axis=(0, 1))
    SY = np.fft.fftshift(np.fft.fft2(Sy))

    Sx = np.pad(Sx, ((0, IMG.shape[0] - 3), (0, IMG.shape[1] - 3)))
    Sx = np.roll(Sx, [-pad, -pad], axis=(0, 1))
    SX = np.fft.fftshift(np.fft.fft2(Sx))

    G1 = IMG * SY
    G2 = IMG * SX
    G = G1 + G2

    fig = px.imshow(np.log(np.abs(G) + 1), color_continuous_scale='gray')
    fig.show()

    T = 5 #prag
    G = G > T
    gprag = np.real(np.fft.ifft2(np.fft.fftshift(G)))
    gprag = gprag[pad:-pad, pad:-pad]

    gprag = ((gprag - gprag.min()) / (gprag.max() - gprag.min())) * 255

    fig = px.imshow(gprag, zmin = 0, zmax = 255, color_continuous_scale='gray')
    fig.show()

#zadatak6_4()

def zadatak7(Q):

    S_shape = (3,3)
    img = io.imread('lena.png')
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    f = np.zeros(img.shape)

    pad_rows = math.floor(S_shape[0] / 2)
    pad_cols = math.floor(S_shape[1] / 2)

    img_p = np.pad(img, ((pad_rows,), (pad_cols,)), mode='symmetric')

    for row in range(f.shape[0]):
        for col in range(f.shape[1]):
            region = img_p[row:row + S_shape[0], col:col + S_shape[1]]

            res = region.sum() **(Q+1) / region.sum ** Q

            f[row, col] = res

    return f

def zadatak7b():

    img_f = zadatak7(3)
    fig = px.imshow(img_f, color_continuous_scale='gray')
    fig.show()

zadatak7b()


