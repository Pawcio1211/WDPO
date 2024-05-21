import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
def empty_callback(value):
    print()

def image1():
    r = 0
    c = 0
    img = cv2.imread('lenna_noise.bmp',cv2.IMREAD_COLOR)
    cv2.namedWindow('img')
    cv2.createTrackbar('R', 'img', 0, 5, empty_callback)
    cv2.createTrackbar('C', 'img', 0, 2, empty_callback)
    while True:
        g = 2 * r + 1
        if c == 0:
            blur = cv2.blur(img, (g, g))
        elif c == 1:
            blur = cv2.GaussianBlur(img, (g, g), 0)
        elif c == 2:
            blur = cv2.medianBlur(img, g)
        cv2.imshow('img', blur)
        r = cv2.getTrackbarPos('R', 'img')
        c = cv2.getTrackbarPos('C', 'img')

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

def image2():
    r1=0
    c1=0
    img2 = cv2.imread('lenna_salt_and_pepper.bmp',cv2.IMREAD_COLOR)
    cv2.namedWindow('img2')
    cv2.createTrackbar('R1', 'img2', 0, 5, empty_callback)
    cv2.createTrackbar('C1', 'img2', 0, 2, empty_callback)
    while True:
        g1=2*r1+1
        #kernel = np.ones((g, g), np.uint8)
        #opening = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)


        if c1 == 0:
            blur2 = cv2.blur(img2, (g1, g1))
        elif c1 == 1:
            blur2 = cv2.GaussianBlur(img2, (g1, g1), 0)
        elif c1 == 2:
            blur2 = cv2.medianBlur(img2, g1)

        cv2.imshow('img2', blur2)

        r1 = cv2.getTrackbarPos('R1', 'img2')
        c1 = cv2.getTrackbarPos('C1', 'img2')

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break



def morphological_operations():
    img = cv2.imread('lenna_noise.bmp', cv2.IMREAD_GRAYSCALE)

    r = 0
    cv2.namedWindow('erosion')
    cv2.namedWindow('dilatation')
    cv2.namedWindow('morphological')
    cv2.createTrackbar('threshold', 'erosion', 0, 255, empty_callback)
    cv2.createTrackbar('R', 'erosion', 0, 5, empty_callback)



    while True:
        threshold = cv2.getTrackbarPos('threshold', 'erosion')

        g = 2 * r + 1
        ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((g, g), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        dilation = cv2.dilate(thresh, kernel, iterations=1)


        cv2.imshow('erosion', erosion)
        cv2.imshow('dilatation', dilation)
        cv2.imshow('morphological', thresh)

        r = cv2.getTrackbarPos('R', 'erosion')

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    cv2.destroyAllWindows()

def scan():
    img = cv2.imread('lenna_noise.bmp', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    cv2.namedWindow('scan')
    licznik = 0
    for i in range(h):
        for j in range(w):
            if j%3== 0:
                img[i ,j] = 255
            if img[i, j] == 255:
                licznik = licznik + 1
    print(licznik/(h*w))
    essa = cv2.blur(img, (3, 3))
    cv2.imshow('scan', essa)
    cv2.waitKey(0)


scan()



////////////////////
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
def empty_callback(value):
    print()

r=1
c=0
t=cv2.THRESH_BINARY
img3 = cv2.imread('kolory.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('lenna_noise.bmp',cv2.IMREAD_COLOR)
img2 = cv2.imread('lenna_salt_and_pepper.bmp',cv2.IMREAD_COLOR)
cv2.namedWindow('img')
cv2.createTrackbar('R', 'img', 0, 5, empty_callback)
cv2.createTrackbar('C', 'img', 0, 2, empty_callback)
while True:
    g=2*r+1
    #kernel = np.ones((g, g), np.uint8)
    #opening = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
    if c==0:
        blur = cv2.blur(img, (g, g))
        blur2 = cv2.blur(img2, (g, g))
    elif c==1:
        blur = cv2.GaussianBlur(img,(g,g),0)
        blur2 = cv2.GaussianBlur(img2, (g, g), 0)
    elif c==2:
        blur = cv2.medianBlur(img, g)
        blur2 = cv2.medianBlur(img2, g)




    cv2.imshow('img', blur2)

    r = cv2.getTrackbarPos('R', 'img')
    c = cv2.getTrackbarPos('C', 'img')

    key_code = cv2.waitKey(10)
    if key_code == 27:
        break

//////////////////////////////////////

import cv2
import numpy as np

img_color=cv2.imread("/home/lab/Downloads/landscape0.jpg", cv2.IMREAD_COLOR)
cv2.imshow("image",img_color)
cv2.waitKey(0)

img_grayscale=cv2.imread("/home/lab/Downloads/landscape0.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("image",img_grayscale)
cv2.waitKey(0)

print(np.shape(img_color))
print(img_color.shape)

px_color=img_color[220,270]

print(f'Pixel value at [220, 270] in scale COLOR: {px_color}')


print(np.shape(img_grayscale))
print(img_grayscale.shape)

px_grayscale=img_grayscale[220,270]

print(f'Pixel value at [220, 270] in scale GRAYSCALE: {px_grayscale}')

mountain=img_color[200:400, 200:400]
cv2.imshow("img_mountain",mountain)
cv2.waitKey(0)

img_color[800:1000, 800:1000] = mountain

cv2.imshow("img_end",img_color)
cv2.waitKey(0)

///////////////////////////
import cv2
import numpy as np

def empty_callback(value):
    pass

img = cv2.imread('krajobraz.jpg',cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('image')

cv2.createTrackbar('treshold', 'image', 0, 255, empty_callback)


while True:

    treshold = cv2.getTrackbarPos('treshold', 'image')

    cv2.imshow('image', img)
    # get current positions of four trackbars
    ret,result=cv2.threshold(img,treshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('image',result)
    key_code = cv2.waitKey(100)
    if key_code == 27:
        break

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
////////////////////////////////