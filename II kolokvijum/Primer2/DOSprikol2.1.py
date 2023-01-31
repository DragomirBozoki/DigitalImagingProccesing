from skimage import io
import plotly.express as px
from skimage import color
from skimage import morphology

img = io.imread('../simboli.png')
""" fig = px.imshow(img)
fig.show() """


img_hsv = color.rgb2hsv(img)
img_hsv[:,:,0][(0.65 < img_hsv[:,:,0]) & (img_hsv[:,:,0] < 0.67)] = 1/6

img_a = color.hsv2rgb(img_hsv)

""" fig = px.imshow(img_a)
fig.show()
 """
B = morphology.square(70)
img_gray = color.rgb2gray(img)
A = img_gray < 0.95
fig = px.imshow(A, color_continuous_scale='gray')
fig.show()

img_m = morphology.opening(A, B)
fig = px.imshow(img_m, color_continuous_scale='gray')
fig.show()

# img_a_hsv = color.rgb2hsv(img_a)
# img_a_hsv[:,:,0][img_m] = 2/3
# img_final = color.hsv2rgb(img_a_hsv)
# fig = px.imshow(img_final, color_continuous_scale='gray')
# fig.show()
