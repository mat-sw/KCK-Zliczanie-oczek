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
    # for img in imgs:
    #     image = img_as_float(io.imread("images\\"+img, as_gray=True))
    #     # prog1 = threshold_isodata(image)
    #     prog2 = threshold_yen(image)
    #
    #     # binary_isodata = closing(image > prog1, square(3))
    #     binary = closing(image > prog2)
    #
    #     # binary = np.logical_or(binary_yen, binary_isodata)
    #     eroded = morphology.erosion(binary, square(8))
    #     clean = morphology.remove_small_objects(eroded, 100)
    #     show_gray(clean)
    #     plt.show()

    for file in os.listdir(".\\images"):
        image = img_as_float(io.imread("images\\"+file, as_gray=True))

        # prog1 = threshold_isodata(image)
        prog2 = threshold_yen(image)

        # binary_isodata = closing(image > prog1, square(3))
        binary = closing(image > prog2)

        # binary = np.logical_or(binary_yen, binary_isodata)
        # fill = ndi.binary_closing(binary)
        # fill = ndi.binary_fill_holes(fill)
        # eroded1 = morphology.erosion(binary, square(10))
        eroded = morphology.erosion(binary, square(25))
        fill = ndi.binary_closing(eroded)
        # dilated = morphology.dilation(eroded, square(10))
        clean = morphology.remove_small_objects(fill, 100)
        plt.imsave("imagescv\\"+file, clean)


        img = cv.imread("imagescv\\"+file)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cimg = img.copy()
        img = cv.medianBlur(img, 5)

        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                                  param1=30, param2=15, minRadius=50, maxRadius=130)
        circles = np.uint16(np.around(circles))

        counter = 0

        # print(circles)

        for i in circles[0, :]:
            counter = counter + 1
            # draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 0, 0), 2)
            # draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        print(counter)

        plt.imshow(img)
        plt.show()

        # # print(len(approx))
        # show_gray(clean)
        # plt.show()
