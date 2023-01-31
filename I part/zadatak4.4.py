import numpy as np
from skimage import io
import plotly.express as px

img = io.imread('lena.png')
IMG = np.fft.fftshift(np.fft.fft2(img, s=(2*img.shape[0], 2*img.shape[1])))

H_size = IMG.shape
u = np.arange(H_size[0]).reshape(-1, 1) - np.floor(H_size[0]/2)
v = np.arange(H_size[1]) - np.floor(H_size[1]/2)
D = np.sqrt(u**2 + v**2)

n = 5
D0 = 60
H_nf = 1 / (1 + (D / D0)**(2*n))
# px.line(x=np.arange(len(H_nf)), y=np.abs(H_nf)[128]).show()

IMG_f = IMG * H_nf
img_f = np.real(np.fft.ifft2(np.fft.ifftshift(IMG_f)))
img_f = img_f[0:img.shape[0], 0:img.shape[1]]


slika = px.imshow(img_f, color_continuous_scale='gray')
slika.show()
