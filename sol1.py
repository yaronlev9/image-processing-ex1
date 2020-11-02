import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from numpy.linalg import linalg
from skimage.color import rgb2gray
import imageio

NUM_OF_PIXELS = 256
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
RGB_IMAGE_SHAPE_SIZE = 3
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))/255

def read_image(filename, representation):
    im = imageio.imread(filename)
    if len(im.shape) == RGB_IMAGE_SHAPE_SIZE and representation == 1:
        img_gray = rgb2gray(im)
        return img_gray
    elif len(im.shape) == RGB_IMAGE_SHAPE_SIZE and representation == 2:
        return im/NUM_OF_PIXELS - 1
    elif len(im.shape) < RGB_IMAGE_SHAPE_SIZE and representation == 1:
        return im / NUM_OF_PIXELS - 1
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

def histogram_equalize(im_orig):
    if len(im_orig.shape) == RGB_IMAGE_SHAPE_SIZE:
        im_YIQ = rgb2yiq(im_orig)
        im_Y = im_YIQ[:,:,0]
        Y, hist, new_hist = calc_histogam((im_Y * NUM_OF_PIXELS - 1).astype(np.int))
        im_YIQ[:,:,0] = Y/NUM_OF_PIXELS - 1
        im_eq = yiq2rgb(im_YIQ)
        return im_eq, hist, new_hist
    else:
        im_grey = (im_orig * NUM_OF_PIXELS - 1).astype(np.int)
        return calc_histogam(im_grey)

def calc_histogam(im):
    hist, bounds = np.histogram(im, NUM_OF_PIXELS, [0, NUM_OF_PIXELS - 1])
    cum_hist = hist.cumsum()
    normalized = cum_hist / cum_hist[-1]
    min_index = np.nonzero(normalized)[0][0]
    norm_hist = np.round((normalized - normalized[min_index]) * (NUM_OF_PIXELS - 1) /
                         (normalized[-1] - normalized[min_index])).astype(np.int)
    Y = norm_hist[im]
    new_hist, bounds = np.histogram(Y, NUM_OF_PIXELS, [0, NUM_OF_PIXELS - 1])
    return Y, hist, new_hist

def quantize (im_orig, n_quant, n_iter):
    if len(im_orig.shape) == RGB_IMAGE_SHAPE_SIZE:
        im_YIQ = rgb2yiq(im_orig)
        im_Y = im_YIQ[:,:,0]
    else:
        im_grey = (im_orig * NUM_OF_PIXELS - 1).astype(np.int)

# calculate z uniformly
# claculate q
#iterate



# im = imageio.imread('externals/low_contrast.jpg')/255
# plt.imshow(im, cmap="gray")
# plt.show()
# im_eq = histogram_equalize(im)[0]
# plt.imshow(im_eq, cmap="gray")
# plt.show()