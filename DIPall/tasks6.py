import plotly.express as px
from skimage import io
import numpy as np
import plotly.graph_objects as go
from scipy import signal as sig
import math
from scipy import ndimage

def zadatak1():

    img = io.imread('lena.jpg')
    img2 = io.imread('lena.jpg')

    IMG = np.fft.fftshift(np.fft.fft2(img))
    IMG2 = np.fft.fftshift(np.fft.fft2(img2))

    IMGA= np.abs(IMG)
    IMGP = np.angle(IMG)

    IMG2A = np.abs(IMG2)
    IMG2P = np.angle(IMG2)

    IMG4 = IMGA*np.exp(1j*IMG2P)
    fig = px.imshow(np.log(np.abs(IMG4)+1), color_continuous_scale='gray')
    fig.show()

    img4 = np.real(np.fft.ifft2(np.fft.fftshift(IMG4)))
    fig = px.imshow(np.log(np.abs(img4) + 1), color_continuous_scale='gray')
    fig.show()

#zadatak1()

def zadatak2(H, D0, n):


    u = np.arange(H[0]).reshape(-1, 1) - np.floor(H[0] / 2)
    v = np.arange(H[1]) - np.floor(H[1] / 2)

    D = np.sqrt(u ** 2 + v ** 2)

    H = 1/(1+(D/D0)**2*n)
    return H

def zadatak2F():

    print("Nisam dodao poprecni presek mrzelo me :D \nbzzzzzzzzzz")
    img = io.imread('lena.jpg')
    H1 = zadatak2((img.shape),50,4)

    IMG = np.fft.fftshift(np.fft.fft2(img))

    IMG = IMG * H1

    fig = px.imshow(np.log(np.abs(IMG)+1), color_continuous_scale='gray')
    fig.show()

    fig.add_trace(go.Surface(z=np.abs(IMG)))
    fig.show()


#zadatak2F()

def zadatak3a(H,D0, C):

    u = np.arange(H[0]).reshape(-1, 1) - np.floor(H[0] / 2)
    v = np.arange(H[1]) - np.floor(H[1] / 2)

    D = np.sqrt(u**2 + v**2)
    Hvf = 1 - np.exp((-D**2)/(2*D0**2))

    if C<0:
        print("C mora bit pozitivno!")
    else:
        HUM = 1+ C*Hvf
        return HUM

def zadatak3():

    img = io.imread('lena.jpg')
    IMG = np.fft.fft2(np.fft.fftshift(img))

    H1 = zadatak3a(img.shape, 50,4)

    IMG = IMG * H1
    fig = px.imshow(np.log(np.abs(IMG) + 1), color_continuous_scale='gray')
    fig.show()

    img = np.real(np.fft.fftshift(np.fft.ifft2(IMG)))
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

#zadatak3()

def zadatak4():

    img = io.imread('lena.jpg')
   # img = [[8, 8, 8, 8, 8], [8, 16, 16, 16, 8], [8, 16, 80, 16, 8], [8, 16, 16, 16, 8], [8, 8, 8, 8, 8]]
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    pad = 1
    img_p = np.pad(img, pad, 'reflect');
    IMG = np.fft.fftshift(np.fft.fft2(img_p))

    # Sobel za y
    S_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Sy_p = np.pad(S_y, ((0, IMG.shape[0] - 3), (0, IMG.shape[1] - 3)))
    Sy_p = np.roll(Sy_p, [-pad, -pad], axis=(0, 1))
    Sy_f = np.fft.fftshift(np.fft.fft2(Sy_p))

    # Sobel za x
    S_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sx_p = np.pad(S_x, ((0, IMG.shape[0] - 3), (0, IMG.shape[1] - 3)))
    Sx_p = np.roll(Sx_p, [-pad, -pad], axis=(0, 1))
    Sx_f = np.fft.fftshift(np.fft.fft2(Sx_p))

    IMG_y = IMG * Sy_f
    IMG_x = IMG * Sx_f

    IMG2 = IMG_y + IMG_x
    img1 = np.real(np.fft.ifft2(np.fft.ifftshift(IMG2)))
    img1 = img1[pad:-pad, pad:-pad]

    img1 = ((img1 - img1.min()) / (img1.max() - img1.min())) * 255

    fig = px.imshow(img1, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    img = io.imread('lena.jpg')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    pad = 1;
    img_p = np.pad(img, pad, 'reflect');
    IMG1 = np.fft.fftshift(np.fft.fft2(img_p))

    IMG_y = IMG1 * Sy_f
    IMG_x = IMG1 * Sx_f

    IMG2 = IMG_y + IMG_x
    img1 = np.real(np.fft.ifft2(np.fft.ifftshift(IMG2)))
    img1 = img1[pad:-pad, pad:-pad]

    fig = px.imshow(img1, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    img = [[-125, 0, 1], [25, 125, 6], [100, -250, 50]]
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()
#zadatak4()


def primer1():

    img = np.zeros((256,256))
    img[0,0] = 1

    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

    img_F = np.fft.fft2(img)

    img_A = np.abs(img_F)
    IMG_F = np.angle(img_F)

    fig = px.imshow(img_A, color_continuous_scale='gray')
    fig.show()
    fig = px.imshow(IMG_F, color_continuous_scale='gray')
    fig.show()

#primer1()

def primer2():

    y = np.arange(0,256,1).reshape(-1,1)
    x = np.arange(0, 256, 1)

    img = np.cos((y+ x) * np.pi / 4)
    fig = px.imshow(img, color_continuous_scale='gray')

    fig.show()

    DFTimge = np.fft.fft2(img)
    DFTimg = np.abs(DFTimge)
    DFTPhase = np.angle(DFTimge)


    fig = px.imshow(DFTimg, color_continuous_scale='gray')
    fig.show()
    fig = px.imshow(DFTPhase, color_continuous_scale='gray')
    fig.show()

#primer2()

def primer3():

    img = io.imread('lena.jpg').astype('uint8')
    imgDFT = np.fft.fft2(img)
    imgniske = np.fft.fftshift(imgDFT)

    fig = px.imshow(np.log(np.abs(imgniske)+1), color_continuous_scale='gray')
    fig.show()

#primer3()

def primer5a(H,D0):

    u = np.arange(H[0]).reshape(-1, 1) - np.floor(H[0]/2)
    v = np.arange(H[1]) - np.floor(H[1]/2)

    D = np.sqrt(u**2 + v**2)

    H = D <= D0
    return H

def primer5b():

    H1 = primer5a((256,256), 50)
    fig = px.imshow(np.abs(H1), color_continuous_scale='gray')
    fig.show()

    fig.add_trace(go.Surface(z = np.abs(H1)))
    fig.show()

#primer5b()

def primer6a(H, D0):

    u = np.arange(H[0]).reshape(-1, 1) - np.floor(H[0] / 2)
    v = np.arange(H[1]) - np.floor(H[1] / 2)

    D = np.sqrt(u ** 2 + v ** 2)

    H = np.exp(((-D**2)/ (2*D0**2)))
    return H

def primer6():

    H1 = primer6a((256,256), 50)
    fig = px.imshow(np.abs(H1), color_continuous_scale='gray')
    fig.show()

#primer6()

def primer7():

    img = np.zeros((256,256))
    img[128, 128] = 1
    img[128 - 64, 128 - 64] = 1
    img[128 + 64, 128 - 64] = 1
    img[128 - 64, 128 + 64] = 1
    img[128 + 64, 128 + 64] = 1


    IMG = np.fft.fftshift(np.fft.fft2(img, s=(2*img.shape[0], 2*img.shape[1])))
    fig = px.imshow(np.log(np.abs(IMG)+1), color_continuous_scale='gray')
    fig.show()

    H1 = primer5a(IMG.shape, 50)
    IMG1 = H1 * IMG

    fig = px.imshow(np.log(np.abs(IMG1)+1), color_continuous_scale='gray')
    fig.show()

    IMGinv = np.fft.ifft2(np.fft.fftshift(IMG1))
    img1 = np.real(IMGinv)
    img1 = img1[0:img.shape[0], 0:img.shape[1]]

    fig = px.imshow(img1, color_continuous_scale='gray')
    fig.show()

#primer7()

def primer8():
    img = np.zeros((256, 256))
    img[128, 128] = 1
    img[128 - 64, 128 - 64] = 1
    img[128 + 64, 128 - 64] = 1
    img[128 - 64, 128 + 64] = 1
    img[128 + 64, 128 + 64] = 1

    H2 = primer6a(img.shape, 100)#Gaus
    IMG2 = img * H2

    fig = px.imshow(IMG2, color_continuous_scale='gray')
    fig.show()

    IMG3 = np.fft.ifft2(np.fft.fftshift(IMG2))
    IMG3 = np.real(IMG3)
    IMG3 = IMG3[0:img.shape[0],0:img.shape[1]]
    fig = px.imshow(IMG3, color_continuous_scale='gray')
    fig.show()

#primer8()

def primer9():

    img = io.imread('lena.jpg')
    #fig = px.imshow(img, color_continuous_scale='gray')
    #fig.show()

    IMG = np.fft.fftshift(np.fft.fft2(img, s=(2*img.shape[0], 2*img.shape[1])))
    H = 1 - primer5a(IMG.shape, 50)

    IMG = IMG * H
    #fig = px.imshow(np.abs(IMG)+1, color_continuous_scale='gray')
    #fig.show()

    img1 = np.real(np.fft.ifft2(np.fft.ifftshift(IMG)))
    img1 = img1[0:img.shape[0], 0:img.shape[1]]
    fig = px.imshow(img1, color_continuous_scale='gray')
    fig.show()

#primer9()

def primer10():
    img = io.imread('lena.jpg')
    Li = [[0,-1,0], [-1,5,-1], [0,-1,0]]
    pad = 1
    img = np.pad(img, pad, 'reflect')
    IMG = np.fft.fftshift(np.fft.fft2(img))

    Li = np.pad(Li,((0,IMG.shape[0]-3),(0,IMG.shape[0]-3)))
    Li = np.roll(Li,[-pad,-pad],axis=(0,1))
    Li = np.fft.fftshift(np.fft.fft2(Li))


    IMG = IMG * Li

    img = np.real(np.fft.ifft2(np.fft.fftshift(IMG)))
    fig = px.imshow(img)
    fig.show()

#primer10()

def main():

    print("1. Zad \n2. Zad \n3. Zad\n4. Zad \n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        zadatak1()
        main()
    elif n == 2:
        img = io.imread('lena.jpg')
        zadatak2(img.shape,50,30)
        print("bzzzzzzzzz :D")
        main()
    elif n ==3:
        print("pokreni rucno, nez parametre")

        main()
    elif n == 4:
        print("tezak je jako")
        zadatak4()
    else:
        print("Greska!")
        main()

main()