import numpy as np
from fir_conv import fir_conv

def sobel_edge(in_img_array: np.ndarray, thres: float ) -> np.ndarray:
    """
    Accept grayscale input image and the threshold thresh. The threshold is 
    the minimum value of the gradient's norm needed to declare a pixel an edge pixel.
    Output is a binary image with values 0 or 1. Edge pixels will be assigned a
    1, pixels that do not belong to an edge get assigned a 0. 

    Parameters:
    -----------
    in_img_array : np.ndarray
        Input grayscale image
    thres : float
        Threshold for edge detection

    Returns:
    --------
    edge_img : np.ndarray
        Binary image with edges (1) and non-edges (0)
    """
    # Define Sobel operators for x and y directions
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Apply Sobel operators using FIR convolution
    grad_x, _ = fir_conv(in_img_array, sobel_x)
    grad_y, _ = fir_conv(in_img_array, sobel_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Apply threshold
    edge_img = np.zeros_like(gradient_magnitude)
    edge_img[gradient_magnitude >= thres] = 1

    return edge_img
