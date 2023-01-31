from matplotlib.pyplot import jet
import plotly.express as px
import numpy as np
from skimage import io
from skimage import color
import matplotlib as mpl
from skimage import exposure


img = io.imread('../peppers.png')
fig = px.imshow(img, color_continuous_scale='gray')
fig.show()


cm_jet = mpl.cm.get_cmap('jet')
im = cm_jet(img)
im = np.uint8(im * 255)[:,:,:3]
print(im.shape)

fig = px.imshow(im)
fig.show()

im_hsv = color.rgb2hsv(im)
im_hsv[:,:,1] = exposure.equalize_hist(im_hsv[:,:,1])
px.imshow(im_hsv).show()
im_hsv[:,:,2] = im_hsv[:,:,2]*1/2
px.imshow(im_hsv).show()
im_hsv[:,:,0][(im_hsv[:,:,0] > 0.65) & (im_hsv[:,:,0] < 0.67)] += 15

im_final = color.hsv2rgb(im_hsv)

fig = px.imshow(im_final)
fig.show()


