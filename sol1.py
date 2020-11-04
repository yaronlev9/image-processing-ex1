import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
from skimage.color import rgb2gray
import imageio

NUM_OF_PIXELS = 256
START_Z_LEVEL = -1
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
RGB_IMAGE_SHAPE_SIZE = 3
x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))/255

def read_image(filename, representation):
    """
    this function recieves a name of an image file, if the image is colored turns it to grayscale if not
    return the image with type float64
    :param filename: the name of the file
    :param representation: if 1 returns and image is colored returns a grayscale image if 2 returns the colored image
    :return: the imange type float 64
    """
    try:
        im = imageio.imread(filename)
    except IOError:
        return
    if representation != 1 and representation != 2:
        return
    if len(im.shape) == RGB_IMAGE_SHAPE_SIZE and representation == 1:
        img_gray = rgb2gray(im)
        return img_gray
    else:
        return im / NUM_OF_PIXELS - 1



def imdisplay(filename, representation):
    """
    displays the image based on the representation if 1 returns grey image if 2 colored image
    :param filename: the name of the file
    :param representation: 1 for grey 2 for colored
    """
    try:
        img = read_image(filename, representation)
    except IOError:
        return
    if representation != 1 and representation != 2:
        return
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()


def rgb2yiq(im_rgb):
    """
    turns rgb image to yiq image
    :param im_rgb: the rgb image
    :return: a new image with yiq
    """
    new_im = np.dot(im_rgb, YIQ_MATRIX.T)
    return new_im


def yiq2rgb(im_yiq):
    """
    turns yiq image to rgb
    :param im_yiq: the yiq image
    :return: the rgb image
    """
    new_im = np.dot(im_yiq, linalg.inv(YIQ_MATRIX).T)
    return new_im


def histogram_equalize(im_orig):
    """
    does histogram equalize on the given image
    :param im_orig: the original image
    :return: the image after histogram eq, the original histogram and the new
    """
    if len(im_orig.shape) == RGB_IMAGE_SHAPE_SIZE:  # checks if the image is rgb type
        im_yiq = rgb2yiq(im_orig)
        im_y = im_yiq[:,:,0]
        y, hist, new_hist = calc_histogam((im_y * (NUM_OF_PIXELS - 1)).astype(np.int))
        im_yiq[:,:,0] = y/(NUM_OF_PIXELS - 1)
        im_eq = yiq2rgb(im_yiq)
        return im_eq, hist, new_hist
    else:
        im_grey = (im_orig * (NUM_OF_PIXELS - 1)).astype(np.int)
        new_im, hist, new_hist = calc_histogam(im_grey)
        return new_im/(NUM_OF_PIXELS - 1), hist, new_hist


def calc_histogam(im):
    """
    gets an image calculates normalized histogram, gets the min index that is not zero for linear stretching
    calculates the mapping histogram and returns a new image equalized.
    :param im: the original image, y or greyscale
    :return: the eq image, old histogram and new histogram
    """
    hist, bounds = np.histogram(im, NUM_OF_PIXELS, [0, NUM_OF_PIXELS - 1])
    cum_hist = hist.cumsum()
    if cum_hist[-1] == 0:
        return im, hist, hist
    normalized = cum_hist / cum_hist[-1]
    min_index = np.nonzero(normalized)[0][0]  # gets the minimal index that is not zero
    norm_hist = np.round((normalized - normalized[min_index]) * (NUM_OF_PIXELS - 1) /
                         (normalized[-1] - normalized[min_index])).astype(np.int)
    y = norm_hist[im]  # does the mapping to a new y or greyscale
    new_hist, bounds = np.histogram(y, NUM_OF_PIXELS, [0, NUM_OF_PIXELS - 1])
    return y, hist, new_hist


def quantize (im_orig, n_quant, n_iter):
    """
    quantize an image
    :param im_orig: the original image
    :param n_quant: the number of q values
    :param n_iter: the number of iterations
    :return: a new image after quantize and the error array
    """
    if n_quant <= 0 or n_iter <= 0:
        return
    if len(im_orig.shape) == RGB_IMAGE_SHAPE_SIZE:  # checks if the image is rgb
        im_yiq = rgb2yiq(im_orig)
        im_y = (im_yiq[:,:,0] * (NUM_OF_PIXELS - 1)).astype(np.int)
        new_im, err = calc_quantize(im_y, n_quant, n_iter)
        im_yiq[:, :, 0] = new_im / (NUM_OF_PIXELS - 1)
        im_quant = yiq2rgb(im_yiq)
        return im_quant, err

    else:
        im_grey = (im_orig * (NUM_OF_PIXELS - 1)).astype(np.int)
        im_quant, err = calc_quantize(im_grey, n_quant, n_iter)
        return im_quant / (NUM_OF_PIXELS - 1), err


def find_uniform_z(hist, n_quant):
    """
    gets an array of uniformed z value based on number of pixels
    :param hist: the image histogram
    :param n_quant: the number of q values
    :return: the z array
    """
    z = np.array([START_Z_LEVEL])
    cum_hist = hist.cumsum()
    if cum_hist[-1] == 0:
        return np.full(n_quant + 1, 0)
    norm_hist = cum_hist / cum_hist[-1]  # calculates the normalized cum histogram
    for i in range(1, n_quant):
        array = np.nonzero(norm_hist >= i/n_quant)  # gets the percentage of pixels that above the percentage of q
        z = np.append(z, array[0][0])  # assign the lowest level that passes the percentage above q percentage
    z = np.append(z, NUM_OF_PIXELS - 1)
    return z


def find_z(q, n_quant):
    """
    gets the z values
    :param q: the q array
    :param n_quant: the number of q
    :return: the z array
    """
    z = np.array([START_Z_LEVEL])
    for i in range(n_quant - 1):
        z = np.append(z, (q[i] + q[i+1])//2)
    z = np.append(z, NUM_OF_PIXELS - 1)
    return z


def find_q (hist, z, n_quant):
    """
    gets the q values
    :param hist: the image histogram
    :param z: the z array
    :param n_quant: the number of q
    :return: the q array
    """
    q = np.array([])
    for i in range(n_quant):
        arr = np.arange(z[i] + 1, z[i+1] + 1)
        arr *= hist[z[i] + 1: z[i+1] + 1]
        divisor_sum = arr.sum()
        divided_sum = hist[z[i] + 1: z[i+1] + 1].sum()
        if divided_sum == 0:
            continue
        q = np.append(q, divisor_sum // divided_sum)
    return q.astype(np.int)


def calc_quantize(im, n_quant, n_iter):
    """
    does the initial values of q and z and does the iterations and checks if the z values change and returns
    the new image and the error array
    :param im: the original image
    :param n_quant: the number of q
    :param n_iter: the number of iterations
    :return: the new image after eq and the array of the errors
    """
    hist, bounds = np.histogram(im, NUM_OF_PIXELS, [0, NUM_OF_PIXELS - 1])
    prev_z = find_uniform_z(hist, n_quant)
    q = find_q(hist, prev_z, n_quant)
    # err = calc_error(hist, prev_z, q)
    err_arr = np.array([], dtype=np.int64)
    for i in range(n_iter):
        z, q, err = make_iteration(hist, q, n_quant)  # does the iteration
        if np.array_equal(prev_z, z):  # checks if the z values changed
            break
        else:
            prev_z = z
            err_arr = np.append(err_arr, err)
    new_im = make_new_im(im, prev_z, q)  # creates the new image based on the z and q values
    return new_im, err_arr


def make_iteration(hist, q, n_quant):
    """
    calculates z and q and the err and returns them
    :param hist: the histogram of the image
    :param q: the q array
    :param n_quant: the number of q
    :return: the z, q, and error values
    """
    z = find_z(q, n_quant)
    q = find_q(hist, z, n_quant)
    err = calc_error(hist, z, q)
    return z, q, err


def calc_error(hist, z, q):
    """
    calculates the error
    :param hist: the image histogram
    :param z: the z array
    :param q: the q array
    :return: the error
    """
    err = 0
    for i in range(q.shape[0]):
        arr = np.arange(z[i] + 1, z[i+1] + 1)
        arr = ((q[i] - arr)**2) * hist[z[i] + 1: z[i+1] + 1]
        err += arr.sum()
    return err


def make_new_im(im, z, q):
    """
    creates a new image based on z and q values
    :param im: the original image
    :param z: the z values
    :param q: the q values
    :return: the new image
    """
    new_hist = np.array([])
    for i in range(q.shape[0]):
        new_hist = np.append(new_hist, np.full((z[i+1] - z[i]), q[i]))  # adds the value of q[i] to the indexes
    new_im = new_hist[im]  # does the mapping to the new images
    return new_im

# im = imageio.imread('externals/monkey.jpg')/255
# plt.imshow(grad, cmap="gray")
# plt.show()
# im_eq, hist, new_hist = histogram_equalize(grad)
# plt.imshow(im_eq, cmap="gray")
# plt.show()
# plt.figure()
# plt.plot(hist.cumsum())
# plt.show()
# plt.figure()
# plt.plot(new_hist.cumsum())
# plt.show()

# im = imageio.imread('externals/monkey.jpg')/255
# im_eq = histogram_equalize(im)[0]
# plt.imshow(im, cmap="gray")
# plt.show()
# im_eq, error = quantize(im, 5, 20)
# plt.imshow(im_eq, cmap="gray")
# plt.show()
# plt.figure()
# plt.plot(error)
# plt.show()

#check initial error