# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(-image, cmap='Greys')
    plt.show()


def show_two(image1, image2):
    plt.subplot(121)
    plt.imshow(-image1, cmap='Greys')

    plt.subplot(122)
    plt.imshow(-image2, cmap='Greys')

    plt.show()


def plot_hist(img):
    plt.hist(img.ravel(), 256, range=(0., 1.), color='red')
    plt.show()


def plot_2img_2hist(image1, image2):

    plt.subplot(221)
    plt.imshow(-image1, cmap='Greys')

    plt.subplot(223)
    plt.hist(image1.ravel(), 256, range=(0., 1.), color='red')

    plt.subplot(222)
    plt.imshow(-image2, cmap='Greys')

    plt.subplot(224)
    plt.hist(image2.ravel(), 256, range=(0., 1.), color='red')

    plt.show()
