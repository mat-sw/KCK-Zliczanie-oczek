from skimage import img_as_float, feature, morphology, filters, img_as_ubyte
import skimage.io as io
from skimage.morphology import square, closing
from skimage.filters import threshold_yen, threshold_isodata, sobel
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, figure, subplot
import os
import numpy as np
from scipy import ndimage as ndi
import cv2 as cv
from PIL import Image

imgs = ["20211121_194435.jpg", "20211121_194320.jpg", "1637665348988.jpg", "1637665349106.jpg"]


def show_gray(img):
    imshow(img, cmap='gray')

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 255), density=True)
    plt.plot(histo)
    plt.xlim(0, 255)


if __name__ == '__main__':
    print("Goodbye World :/")

    for file in os.listdir(".\\images"):
        # pass
    # for img in imgs:
    #     image = img_as_float(io.imread("images\\"+file, as_gray=True))
    #     # image = img_as_float(io.imread("images\\"+img, as_gray=True))
    #     prog1 = threshold_isodata(image)
    #     prog2 = threshold_yen(image)
    #
    #     binary_isodata = closing(image > prog1, square(3))
    #     binary_yen = closing(image > prog2)
    #
    #     binary = np.logical_or(binary_yen, binary_isodata)
    #     # fill = ndi.binary_closing(binary)
    #     # fill = ndi.binary_fill_holes(fill)
    #     eroded = morphology.erosion(binary, square(13))
    #     clean = morphology.remove_small_objects(eroded, 2000)
    #     plt.imsave("imagescv\\"+file, clean)

        img = cv.imread("images\\"+file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cimg = img.copy()
        # converting image into grayscale image
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.medianBlur(gray, 5)
        circles = cv.HoughCircles(image=img, method=cv.HOUGH_GRADIENT, dp=0.9,
                                   minDist=80, param1=110, param2=39, maxRadius=70)

        for co, i in enumerate(circles[0, :], start=1):
            # draw the outer circle in green
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle in red
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        plt.imshow(img)
        plt.show()
        # img = cv.resize(img, (960, 640))
        # cv.imshow('shapes', img)
        # # print(len(approx))
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # show_gray(clean)
        # plt.show()
