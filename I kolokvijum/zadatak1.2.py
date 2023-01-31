from skimage import io
import numpy as np
import plotly.express as px
from skimage.transform import resize, rescale
from skimage import color


def kolaz_1(slika, d):

    N, M = slika.shape
    slika1 = slika[0:round(0.5*N), 0:round(0.5*M)]
    slika4 = slika[round(0.5*N):N, round(0.5*M):M]

    slika1_p = np.pad(slika1, pad_width=d, mode='constant', constant_values=0)
    slika4_p = np.pad(slika4, pad_width=d, mode='constant', constant_values=255)
    slika1 = resize(slika1_p, slika1.shape, preserve_range=True)
    slika4 = resize(slika4_p, slika4.shape, preserve_range=True)

    slika[0:round(0.5*N), 0:round(0.5*M)] = slika4
    slika[round(0.5 * N):N, round(0.5 * M):M] = slika1



    img = io.imread('paketici1.png')
    px.imshow(img, color_continuous_scale='gray').show()
    img1 = kolaz_1(img, 10)
    px.imshow(img1, color_continuous_scale='gray').show()


def zadatak1(d):

    img = io.imread('paketici1.png')
    N = img.shape[0]
    M = img.shape[1]

    img1 = img[0:round(N/2), 0:round(M/2)]
    img2 = img[round(N/2):round(N), round(M/2):round(M)]

    img1_p = np.pad(img1, pad_width=d, mode='constant', constant_values=0)
    img2_p = np.pad(img2, pad_width=d, mode='constant',constant_values=255)

    img1 = resize(img1_p, img1.shape, preserve_range=True)
    img2 = resize(img2_p, img2.shape, preserve_range=True)

    img[0:round(0.5 * N), 0:round(0.5 * M)] = img1
    img[round(0.5 * N):N, round(0.5 * M):M] = img2

    fig = px.imshow(img, color_continuous_scale='gray')
    fig.show()

#zadatak1(25)

def zadatak12():

    im = io.imread('jelka.png')
    im1 = np.round(rescale(im, 1 / 2, order=0, preserve_range=True)).astype('uint8')
    im2 = np.round(rescale(im, 1 / 4, order=0, preserve_range=True)).astype('uint8')

    N, M = im.shape
    pozadina = np.ones(shape=(round(0.75 * N), M)) * im[5, 5]
    pozadina[0:round(141 / 4), round(3 * 213 / 8):round(5 * 213 / 8)] = im2
    pozadina[round(141 / 4):(round(141 / 4) + round(141 / 2)), round(213 / 4):round(3 * 213 / 4 - 1)] = im1


    slika = np.concatenate((pozadina, im), axis=0).astype('uint8')
    io.imsave('jelka_2.png', slika)
    px.imshow(slika, color_continuous_scale='gray').show()

zadatak12()