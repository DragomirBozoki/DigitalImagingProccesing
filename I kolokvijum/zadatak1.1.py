from skimage import io
from skimage.transform import rescale
import numpy as np
import plotly.express as px

im = io.imread('jelka.png')
im1 = np.round(rescale(im, 1/2, order=0, preserve_range=True)).astype('uint8')
im2 = np.round(rescale(im, 1/4, order=0, preserve_range=True)).astype('uint8')

N, M = im.shape
pozadina = np.ones(shape=(round(0.75*N), M)) * im[5, 5]
pozadina[0:round(141/4), round(3*213/8):round(5*213/8)] = im2
pozadina[round(141/4):(round(141/4)+round(141/2)), round(213/4):round(3*213/4-1)] = im1

# print(im.shape, im1.shape, im2.shape)
#
# pad1 = int((im.shape[0] - im1.shape[0]) / 2)
# pad11 = int(pad1+1)
# pad2 = int((im.shape[0] - im2.shape[0]) / 2)
# print(pad1, pad2, pad11)
#
# im3 = np.pad(im1, ((pad1, pad11), (0,)), mode='constant', constant_values=im[0, 0])
# im4 = np.pad(im2, ((pad2,), (0,)), mode='constant', constant_values=im[0, 0])
#
# print(im.shape, im3.shape, im4.shape)

# pozadina = np.ones(im.shape) * im[0, 0]
# centralni = pozadina.shape[0] / 2
# pola1 = im1.shape[0] / 2
# pola2 = im2.shape[0] / 2
# pozadina[centralni-pola1:centralni+pola1+1] = im1
# pozadina[centralni-pola2:centralni+pola2+1] = im2
slika = np.concatenate((pozadina, im), axis=0).astype('uint8')
io.imsave('jelka_2.png', slika)
px.imshow(slika, color_continuous_scale='gray').show()
