import numpy as np
import matplotlib.pyplot as plt
# from scipy import misc
from numpy.linalg import linalg
from skimage.color import rgb2gray
import imageio
import skimage.color

NUM_OF_PIXELS = 255
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])

x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))/255

def read_image(filename, representation):
    im = imageio.imread(filename)
    if len(im.shape) == 3 and representation == 1:
        img_gray = rgb2gray(im)
        return img_gray
    elif len(im.shape) == 3 and representation == 2:
        return im/NUM_OF_PIXELS
    elif len(im.shape) < 3 and representation == 1:
        return im / NUM_OF_PIXELS
    else:
        raise()

def imdisplay(filename, representation):
    try:
        img = read_image(filename, representation)
    except:
        return 'wrong input'
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()

def rgb2yiq(imRGB):
    newIm = np.dot(imRGB, YIQ_MATRIX.T)
    return newIm

def yiq2rgb(imYIQ):
    newIm = np.dot(imYIQ, linalg.inv(YIQ_MATRIX).T)
    return newIm


