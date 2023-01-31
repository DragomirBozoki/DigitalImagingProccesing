from skimage import io
import plotly.express as px

A = io.imread('lena.png')
A = A.astype('double')
fig = px.imshow(A, color_continuous_scale='gray')
fig.show()

N, M = A.shape
B = A[round(0.2*N):round(0.8*N), round(0.2*M):round(0.8*M)]
B = B[::-1]
B = (B - B.min()) / (B.max() - B.min())
fig = px.imshow(B, color_continuous_scale='gray')
fig.show()
