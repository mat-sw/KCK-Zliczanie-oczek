from skimage import img_as_float, feature, morphology, img_as_ubyte, img_as_int
import skimage.io as io
from skimage.morphology import square, closing
from skimage.filters import threshold_yen, threshold_isodata
from skimage.restoration import denoise_bilateral
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

    for file in os.listdir(".\\kck_compressed"):
        image = img_as_float(io.imread("kck_compressed\\"+file, as_gray=True))

        denoised = denoise_bilateral(image)
        prog = threshold_yen(denoised)
        binary = closing(image > (prog*1.32))

        clean0 = morphology.remove_small_objects(binary, 10)
        clean50 = morphology.remove_small_objects(clean0, 50)
        clean100 = morphology.remove_small_objects(clean50, 100)
        eroded = morphology.erosion(clean100, square(15))
        fill = ndi.binary_closing(eroded)
        fill = (255-fill)
        # figure(figsize=(8, 6))
        # subplot(1,2,1)
        plt.imsave("imagescv\\"+file, fill)

        img = cv.imread("imagescv\\"+file, 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 100
        params.maxThreshold = 150

        # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 50
        # params.maxArea = 200

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 1
        # # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.75
        params.maxConvexity = 1

        # Filter by Inertio
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        # detector = cv.SimpleBlobDetector_create(params)
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv.SimpleBlobDetector(params)
        else :
            detector = cv.SimpleBlobDetector_create(params)

        # Detect blobs.
        blobs = detector.detect(img)
        img_with_blobs = cv.drawKeypoints(img, blobs, np.array([]), (0,0,0),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print(blobs.BLOB_LABEL_COUNT)
        imshow(img_with_blobs)
        plt.show()
        # figure(figsize=(8, 6))
        # subplot(1,2,1)
        # subplot(1,2,2)
        # plot_hist(img)
