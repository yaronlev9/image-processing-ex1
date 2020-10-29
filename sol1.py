import numpy as np
import matplotlib.pyplot as plt
# from scipy import misc
from skimage.color import rgb2gray
import imageio

NUM_OF_PIXELS = 255

# x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
# grad = np.tile(x,(256,1))
# print(x)
# imageio.imwrite("before.jpg", grad)

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

# imdisplay('low_contrast.jpg', 1)

