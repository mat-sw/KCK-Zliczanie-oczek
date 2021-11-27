from skimage import img_as_float, feature, morphology, filters, img_as_ubyte
import skimage.io as io
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, figure, subplot
import os
import numpy as np

imgs = []

def show_gray(img):
    imshow(img, cmap='gray')

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 255), density=True)
    plt.plot(histo)
    plt.xlim(0, 255)


if __name__ == '__main__':
    print("Goodbye World :/")

    # filepath = ".\\images"
    for file in os.listdir(".\\images"):
        image = io.imread("images\\"+file, as_gray='true')
        # image = feature.canny(image, sigma=2)
        figure()
        # image = (image > 50) * 255
        # image = np.uint8(image)
        subplot(2,2,1)
        show_gray(image)
        subplot(2,2,2)
        plot_hist(image)
        # plt.show()
        MIN = 45 / 255
        MAX = 150 / 255
        # nie wiem czy chcemy sie bawic w takie wycinanie kolorow, spojrz image-processing.html to jest normalizacja -> In[23]
        norm = (image - MIN) / (MAX - MIN)
        norm[norm > 1] = 1
        norm[norm < 0] = 0
        subplot(2,2,3)
        show_gray(norm)
        subplot(2,2,4)
        plot_hist(norm)
        plt.show()