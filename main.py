from skimage import img_as_float, morphology
import skimage.io as io
from skimage.morphology import square, closing
from skimage.filters import threshold_yen
from skimage.restoration import denoise_bilateral
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import os
import numpy as np
from scipy import ndimage as ndi
import cv2 as cv

imgs = ["20211207_103246.jpg", "20211121_194323.jpg", "20211121_193341.jpg", "1637665348825.jpg", "20211206_132932.jpg"]

if __name__ == '__main__':
    dict = {}
    dobre = []
    zle = []
    f = open("lista.txt", "r")
    for line in f:
        line1 = line.split(" ")
        dict[line1[0]] = int(line1[1])
    # print(dict)

    for file in imgs:
    # for file in os.listdir(".\\images"):
        image = img_as_float(io.imread("images\\"+file, as_gray=True))

        denoised = denoise_bilateral(image)
        prog = threshold_yen(denoised)
        binary = closing(image > (prog*1.3))

        clean0 = morphology.remove_small_objects(binary, 10)
        clean50 = morphology.remove_small_objects(clean0, 50)
        clean100 = morphology.remove_small_objects(clean50, 100)
        eroded = morphology.erosion(clean100, square(12))
        fill = ndi.binary_closing(eroded)
        # fill = (255-fill)
        # figure(figsize=(8, 6))
        # subplot(1,2,1)
        plt.imsave("imagescv\\"+file, fill)

        img = cv.imread("imagescv\\"+file, 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # SimpleBlobDetector - PARAMETRY
        params = cv.SimpleBlobDetector_Params()

        # threshold
        # params.minThreshold = 100
        # params.maxThreshold = 150

        # area
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 50000

        # circularity
        params.filterByCircularity = True
        params.minCircularity = 0.75
        params.maxCircularity = 1

        # convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.maxConvexity = 1

        # color
        # params.filterByColor = True
        # params.blobColor = 255

        # interio
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        # detektor
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv.SimpleBlobDetector(params)
        else :
            detector = cv.SimpleBlobDetector_create(params)

        # zliczanie
        blobs = detector.detect(img)
        img_with_blobs = cv.drawKeypoints(img, blobs, np.array([]), (0,255,255),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img = cv.bitwise_and(img,img_with_blobs)

        correctness = ""
        if dict[file] == len(blobs):
            dobre.append(file)
            correctness = "+"
        else:
            zle.append(file)
            correctness = "-"

        print("{} : {}   {}".format(file, len(blobs), correctness))
        imshow(img)
        plt.show()
    print("--------------------------------")
    print("PODSUMOWANIE")
    print("DOBRZE:", len(dobre))
    print("ZLE:", len(zle))
    print("SKUTECZNOSC:", (len(dobre)/(len(dobre)+len(zle)))*100, "%")
