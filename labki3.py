#labki3
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

////////////////////////////////
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

///////////////////////////////
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread('lenna_salt_and_pepper.bmp')
kernel = np.ones((5,5),np.float32)/25
filtr_usredniajacy=cv.filter2D(img,-1,kernel)
filtr_blur=cv.blur(img,(5,5))
filtr_gaussa=cv.GaussianBlur(img,(5,5),0)
filtr_medianowy=cv.medianBlur(img,5)

plt.subplot(331),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(filtr_usredniajacy),plt.title('Usredniajacy')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(filtr_blur),plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(filtr_gaussa),plt.title('Gauss')
plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(filtr_medianowy),plt.title('Medianowy')
plt.xticks([]), plt.yticks([])
plt.show()