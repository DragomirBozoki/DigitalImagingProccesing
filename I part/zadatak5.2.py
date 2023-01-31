from skimage import io
import numpy as np
import plotly.express as px

img = io.imread('deda_mraz.png')

H_size = img.shape #treba da je shape od transformisane slike
n = 5
D0 = 100

u = np.arange(H_size[0]).reshape(-1, 1) - np.floor(H_size[0]/2)
v = np.arange(H_size[1]) - np.floor(H_size[1]/2)
D = np.sqrt(u**2 + v**2)
h = 1 / (1 + (D/D0)**(2*n))

IMG = np.fft.fftshift(np.fft.fft2(img, s=(2*img.shape[0], 2*img.shape[0])))
amp = px.imshow(np.log(np.abs(IMG) + 1), color_continuous_scale='gray')
amp.show()

H = np.fft.fftshift(np.fft.fft2(h, s=(2*img.shape[0], 2*img.shape[0])))
IMG_f = IMG * H
AMP = px.imshow(np.log(np.abs(IMG_f) + 1), color_continuous_scale='gray')
AMP.show()

img_f = np.real(np.fft.fft2(np.fft.fftshift(IMG_f)))
px.imshow(img_f, color_continuous_scale='gray').show()
