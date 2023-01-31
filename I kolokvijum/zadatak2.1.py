from skimage import io
import numpy as np
import plotly.express as px

im = io.imread('presents.png')
px.imshow(im, zmin=0, zmax=255, color_continuous_scale='gray').show()

segment1 = im[0:100]
segment2 = im[100:200]
segment3 = im[200:300]

LUT = []
for s in range(1, 4):
    lut1 = np.linspace(0, 200, s*50+1)
    lut2 = np.linspace(200, 255, 256-s*50)
    lut = np.concatenate((lut1[:-1], lut2), axis=0)
    print(len(lut))
    LUT.append(lut)

im_kor1 = LUT[0][segment1].astype('uint8')
im_kor2 = LUT[1][segment2].astype('uint8')
im_kor3 = LUT[2][segment3].astype('uint8')

im_kor = np.concatenate((im_kor1, im_kor2, im_kor3), axis=0)
px.imshow(im_kor, zmin=0, zmax=255, color_continuous_scale='gray').show()
io.imsave('presents_korig.png', im_kor)
