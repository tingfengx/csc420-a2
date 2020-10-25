"""
Module from homework one. 

Main function: convolution function, and calculating gradient magnitudes
"""

import numpy as np

def get_gaussian_kernel(size = (3, 3), sigma=1.):
    # unpack two components of size
    m, n = size
    # cast them to int, avoid float error
    m, n = int(m), int(n)
    # linspace to expand on coordinates for each dimension
    y = np.linspace(-m / 2, m / 2, m)
    x = np.linspace(-n / 2, n / 2, n)
    
    # meshgrid so we can evluate all positions at once
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
    
    # normalize and return
    return kernel / np.sum(kernel)


def pad_image(image, pad_sizex, pad_sizey):
    """
    input (m, n) image
    input pad_sizex, padding on left and right of the image
    input pad_sizey, padding on top and bottom of the image
    output (m + 2pad_sizey) x (n + 2pad_sizex) image, all padded with zero
    """
    return np.pad(image, [(pad_sizey, pad_sizey), (pad_sizex, pad_sizex)])


def flip_both_sides(kernel):
    return np.fliplr(np.flipud(kernel))


def handle_kernel_even(kernel):
    k, k_ = kernel.shape
    
    # handle k is even case
    if k % 2 == 0:
        # log the info
        print(f"detected kernel has even y-direction size {k}, padding with zero at top to make y-size {k + 1}")
        # make the padding
        kernel = np.pad(kernel, [(1, 0), (0, 0)])
        
    # handle k_ is even case
    if k_ % 2 == 0:
        # log the info
        print(f"detected kernel has even x-direction size {k_}, padding with zero at left to make x-size {k_ + 1}")
        # make the padding
        kernel = np.pad(kernel, [(0, 0), (1, 0)])
        
    return kernel


def conv_full(image, kernel):
    """
    convolves kernel with image, padded so that output size = image size
    
    image I, size m x n numpy array
    kernel,  size k x k_ numpy array
    outputs  size m x n numpy array
    """
    # handle the case of even sized kernels 
    kernel = handle_kernel_even(kernel)
    
    # store padded kernel, assert should be odd sizes
    k, k_ = kernel.shape
    assert k % 2 == 1 and k_ % 2 == 1, "something went wrong! should already been handled!"
    
    # flip the kernel along both axies
    kernel = flip_both_sides(kernel)
    
    # zeros for storting conv results
    result = np.zeros(image.shape)
    
    # padding sizes for image, calculated based on the size of the kernel
    pad_sidey = int((k - 1) / 2)
    pad_sidex = int((k_ - 1) / 2)
    
    padded_image = pad_image(image, pad_sidex, pad_sidey)
    padded_m, padded_n = padded_image.shape
    # loop over vertical direction
    for j in range(pad_sidey, padded_m - pad_sidey):
        # loop over horizontal direction
        for i in range(pad_sidex, padded_n - pad_sidex):
            # selected matrix box to perform matrix dot with flipped kernel
            selected = padded_image[j - pad_sidey: j + pad_sidey + 1, 
                                    i - pad_sidex: i + pad_sidex + 1]
            ypos = j - pad_sidey
            xpos = i - pad_sidex
            # inner product between selected box, then sum and store
            result[ypos, xpos] = np.sum(selected * kernel)
    return result


def calc_grad_magnitude(
    image,
    gx = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]),
    gy = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])):
    """
    calculate gradient magnitudes, using gx and gy as finite difference 
    gradeint convolution operators. Default Sobel Filters
    """
    return np.sqrt(
        conv_full(image, gx) ** 2 + conv_full(image, gy) ** 2
    )