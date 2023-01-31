import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# STISNI RUN DA BI POKRENUO MAIN

def zadatak1():

    #Njegovi fajlovi su se pobrisali, naravno, jer ne zna da ih sacuva na drajvu
    # a na internetu se nigde ne mogu naci uint16 slike, tako da zmin i zmax samo treba staviti na 3000 i 4000

    img = io.imread('gray.jpg')
    fig = px.imshow(img , zmin = 125, zmax = 200, color_continuous_scale= 'gray')

    print(img.dtype)

    fig.show()

    img = io.imread('gray.jpg')
    fig2 = px.imshow(img , zmin = 20, zmax = 100, color_continuous_scale= 'gray')

    fig2.show()
#zadatak1()

def zadatak2(): #setWindowsLevel

    img = io.imread('gray.jpg')

    L = round(img.max()/2) #Taking the middle of 2^8 range
    print(L) #treba 2^16 al ja radim sa 2^8 slikom
    W = 30 #on uzima 3000

    fig = px.imshow(img, zmin = L - W/2, zmax = L + W/2, color_continuous_scale='gray')
    fig.show()

    #pod b sve isto samo sa durgim vrednostima

#zadatak2()

def zadatak3():

    img = io.imread('black.jpg')
    print(img.dtype)

    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    N = img.shape[0]
    M = img.shape[1]

    img2 = img.copy()
    img2 = img2[round(N * 0.30):round(N * 0.60),
           round(M * 0.30):round(M * 0.60)]

    fig = px.imshow(img2, zmin=0,zmax=255, color_continuous_scale='gray')
    fig.show()



#zadatak3()

def zadatak4_a():
    img = io.imread('chad.jpg')
    # normalizerange()

    img_norm = (img - img.min()) / (img.max() - img.min())
    print(img_norm)
    print("\n")

    return img_norm

zadatak4_a()

def zadatak4(img_norm):

    mina = eval(input("Min vrednost slike: "))
    maxa = eval(input("Max vrednost slike: "))
    img = io.imread(img_norm)
    fig = px.imshow(img, zmin = mina, zmax= maxa, color_continuous_scale='gray' )

    fig.show()

#zadatak4('chad.jpg')

def zadatak5():

    #Treba sa np.arange flipati Y osu tipa np.arange(Y[-1]:Y[0]) al stvarno mi se ne da to brljavit po netu

    img = io.imread('tam.jpg')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')

    img_fliplr = np.fliplr(img)


    fig = px.imshow(img_fliplr, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.show()

#zadatak5()

def primer1():

    img = io.imread('lena.jpg')
    fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale= 'gray')

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()
    print(img)

#primer1()

def primer2():

    img = io.imread('baboon.png')
    fig = px.imshow(img, zmin = 0, zmax = 255, color_continuous_scale= 'gray')

    img2 = img + 100
    fig = px.imshow(img2, zmin = 0, zmax= 255, color_continuous_scale='gray')

    img3 = img - 100
    fig = px.imshow(img3, zmin=0, zmax=255, color_continuous_scale='gray')



    fig.show()


#primer2()

def primer3():

    img = io.imread('zelda.png')
    fig = px.imshow(img, zmin=0,zmax=255, color_continuous_scale='gray')

    fig.show()

    img_norm = (img - img.min()) / (img.max() - img.min())

    fig = px.imshow(img_norm, zmin = 0, zmax = 1, color_continuous_scale='gray')

    fig.show()

#primer3()

def primer4():

    img = io.imread('stevo.png')
    img2 = io.imsave('stevokanta.png', img)
    fig = px.imshow(img, zmin = 0, zmax = 255, color_continuous_scale='gray')

    fig.show()

    img2 = io.imread('stevokanta.png')
    fig2 = px.imshow(img2, color_continuous_scale= 'gray')

    fig2.show()

#primer4()

def primer5():

    img = io.imread('jazvezda.jpg')
    fig = px.imshow(img, zmin = 0 , zmax = 255, color_continuous_scale='gray')

    #fig.show()

    N = img.shape[0]
    M = img.shape[1]

    img_c = img.copy()
    # We need to copy IMG into new variable so new parameters will not mess with original image

    img2 = img_c[round(N*0.25):round(N*0.75),
           round(M*0.25):round(M*0.75)]

    fig2 = px.imshow(img2, color_continuous_scale='gray')

    #fig2.show()

    plt.subplot(1, 2, 1)
    plt.imshow(img_c, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.show()


#primer5()

def primer6():

    img = io.imread('chad.jpg')
    #fig = px.imshow(img, zmin=0,zmax=255, color_continuous_scale='gray')

    print("======")
    img_UD = np.flipud(img)
    img_fliplr = np.fliplr(img)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_UD)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(img_fliplr)

#primer6()

def main():

    print("1. Zad \n2. Zad \n3. Zad \n4. zad \n5. Zad \n")

    n = eval(input("Unesite broj zadatka: "))

    if n == 1:
        zadatak1()
        main()
    elif n == 2:
        zadatak2()
        main()
    elif n ==3:
        zadatak3()
        main()
    elif n ==4:
       zadatak4('chad.jpg')
    elif n ==5:
        zadatak5()
        main()
    else:
        print("Greska!")
        main()

main()
