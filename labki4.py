
import cv2
import numpy as np


def apply_kuwahara(image: np.ndarray, window_size: int):
    result=np.zeros_like(image)
    for y in range(image.shape[0]-window_size):
        for x in range(image.shape[1]-window_size):
            window=image[y:y+window_size,x:x+window_size]
            half_window=window_size//2
            region1=window[0:half_window+1, 0:half_window+1]
            region2=window[0:half_window+1, half_window:window_size]
            region3=window[half_window:window_size,half_window:window_size]
            region4=window[half_window:window_size,0:half_window+1]

            best_mean,best_std=cv2.meanStdDev(region1)
            for region in (region2, region3, region4):
                mean, std=cv2.meanStdDev(region)
                if std<best_std:
                    best_mean=mean
                    best_std=std

            result[y+window_size//2,x+window_size//2]=best_mean
    return result


def main():
    image=cv2.imread('piesek.jpg',cv2.IMREAD_GRAYSCALE)

    cv2.imshow('image',image)
    result=apply_kuwahara(image, 5)
    cv2.imshow('result',result)
    cv2.waitKey()

main()

niemoje4:
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# mouse callback function
counter1 = 0
counter2 = 0
i = 0
out = np.hstack((300,300))
xarr = [0, 0, 0, 0]
yarr = [0, 0, 0, 0]
img = np.zeros((512, 512, 3), np.uint8)
img2 = cv.imread('road.jpg')
cv.namedWindow('image')
dim= [int(img2.shape[1]*0.5),int(img2.shape[0]*0.5)]
img2=cv.resize(img2,dim)


def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),20,(255,0,0),-1)
        param["counter1"] = param["counter1"] + 1
        cv.putText(img,str(param["counter1"]),(x-10,y+10),cv.FONT_ITALIC,1,(0,0,255))
    if event == cv.EVENT_MBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+50,y+50),(100,100,0),-1)
        param["counter2"] = param["counter2"] + 1
        cv.putText(img,str(param["counter2"]),(x+18,y+35),cv.FONT_ITALIC,1,(0,0,255))

def rysowanie():
    cv.setMouseCallback('image',draw_circle, {'counter1': counter1,'counter2': counter2})
    while(1):
        cv.imshow('image',img)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()

def prostowanie(event,x,y,flags,param):
    global i, out
    if event == cv.EVENT_LBUTTONDBLCLK:
        param['xarr'][i] = x
        param['yarr'][i] = y
        i = i + 1

    if i > 3:
        pts1 = np.float32([[param['xarr'][0],param['yarr'][0]],[param['xarr'][1],param['yarr'][1]],[param['xarr'][2],param['yarr'][2]],[param['xarr'][3],param['yarr'][3]]])
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv.getPerspectiveTransform(pts1,pts2)
        dst = cv.warpPerspective(img2,M,(300,300))
        # plt.subplot(121),plt.imshow(img2),plt.title('Input')
        # plt.subplot(122),plt.imshow(dst),plt.title('Output')
        # plt.show()
        out = np.hstack([cv.resize(img2,(300,300)), dst])
        # cv.imshow('image', out)

def rysowanie2():
    cv.setMouseCallback('image', prostowanie, {'xarr': xarr, 'yarr': yarr, 'out': out})
    while(1):
        if i>3:
            cv.imshow('image', out)
        else:
            cv.imshow('image', img2)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()


def hist():
    plt.hist(img2.ravel(), 256, [0, 256]);
    cv.imshow('image', img2)
    plt.show()

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

hist()


////////////////