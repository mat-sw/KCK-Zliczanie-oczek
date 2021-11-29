from skimage import img_as_float, feature, morphology, filters, img_as_ubyte
import skimage.io as io
from skimage.morphology import square, closing
from skimage.filters import threshold_yen, threshold_isodata, sobel
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, figure, subplot
import os
import numpy as np
from scipy import ndimage as ndi

imgs = ["20211121_194435.jpg", "20211121_194320.jpg", "1637665348988.jpg", "1637665349106.jpg"]

def show_gray(img):
    imshow(img, cmap='gray')

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 255), density=True)
    plt.plot(histo)
    plt.xlim(0, 255)


if __name__ == '__main__':
    print("Goodbye World :/")

    for file in os.listdir(".\\images"):
    #     pass
    # for img in imgs:
        image = img_as_float(io.imread("images\\"+file, as_gray=True))
        # image = img_as_float(io.imread("images\\"+img, as_gray=True))
        prog1 = threshold_isodata(image)
        prog2 = threshold_yen(image)

        binary_isodata = closing(image > prog1, square(3))
        binary_yen = closing(image > prog2)

        binary = np.logical_or(binary_yen, binary_isodata)
        # fill = ndi.binary_closing(binary)
        # fill = ndi.binary_fill_holes(fill)
        eroded = morphology.erosion(binary, square(13))
        clean = morphology.remove_small_objects(eroded, 500)
        show_gray(clean)
        plt.show()