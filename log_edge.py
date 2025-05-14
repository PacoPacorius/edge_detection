import numpy as np
from fir_conv import fir_conv

def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    """
    Accepts the input grayscale image in_img_array and outputs resulting
    binary image with values 0 or 1. Edge pixels will be assigned a
    1, pixels that do not belong to an edge get assigned a 0.
    This function has to call the FIR convolution function.
    Edge detection using Laplacian of Gaussian (LoG) operator.

    Parameters:
    -----------
    in_img_array : np.ndarray
        Input grayscale image

    Returns:
    --------
    edge_img : np.ndarray
        Binary image with edges (1) and non-edges (0)
    """
    # Define Laplacian of Gaussian (LoG) kernel
    # We'll use a 5x5 LoG kernel with sigma ≈ 1.0
    log_kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1,-2, -1, 0],
        [-1,-2,16, -2, -1],
        [0, -1,-2, -1, 0],
        [0, 0, -1, 0, 0]
    ]) / 16.0  # Normalizing factor

    # Apply LoG filter using FIR convolution
    filtered_img, _ = fir_conv(in_img_array, log_kernel)

    edge_img = np.zeros_like(filtered_img)

    gradient_x = np.zeros_like(filtered_img)
    gradient_y = np.zeros_like(filtered_img)
    
    # Simple gradient calculation using central differences
    gradient_x[:, 1:-1] = (filtered_img[:, 2:] - filtered_img[:, :-2]) / 2.0
    gradient_y[1:-1, :] = (filtered_img[2:, :] - filtered_img[:-2, :]) / 2.0
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # this needs to be much better. try to make a binary tree of decisions.
    # Check for zero crossings between adjacent pixels
    rows, cols = filtered_img.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            center = filtered_img[i, j]
            neighbor_positions = [
                (i+1, j), (i-1, j), (i, j+1), (i, j-1),
                (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)
            ]


            # If the center pixel has a different sign from any neighbor, it's a zero crossing
            for ni, nj in neighbor_positions:
                neighbor = filtered_img[ni, nj]
                # If sign change detected and gradient magnitude is above threshold
                if (center * neighbor < 0) and gradient_magnitude[i, j] > 10:
                    edge_img[i, j] = 1
                    break

    return edge_img
