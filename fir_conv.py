import numpy as np

def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: np.ndarray, 
             mask_origin: np.ndarray) -> np.ndarray, np.ndarray:
    """
    Accept a grayscale image with in_img_array. Perform FIR convolution with h. 
    Optionally, 
    define beginning of image coords with in_origin and convolution mask 
    coords with mask_origin. Output the image-result of convolution and 
    optionally the beginning of the output image's coords.
    Perform FIR convolution of a grayscale image with a mask.

    Parameters:
    -----------
    in_img_array : np.ndarray
        Input grayscale image
    h : np.ndarray
        Convolution mask
    in_origin : np.ndarray, optional
        Beginning coordinates of the image
    mask_origin : np.ndarray, optional
        Beginning coordinates of the convolution mask

    Returns:
    --------
    out_img : np.ndarray
        Result of convolution
    out_origin : np.ndarray, optional
        Beginning coordinates of the output image
    """
    # If origins are not provided, use defaults (center of mask)
    if in_origin is None:
        in_origin = np.array([0, 0])

    if mask_origin is None:
        mask_origin = np.array([h.shape[0]//2, h.shape[1]//2])

    # Calculate output image dimensions
    out_height = in_img_array.shape[0] - h.shape[0] + 1
    out_width = in_img_array.shape[1] - h.shape[1] + 1

    # Initialize output image
    out_img = np.zeros((out_height, out_width))

    # Calculate output origin
    out_origin = in_origin + mask_origin

    # Perform convolution
    for i in range(out_height):
        for j in range(out_width):
            # Extract region of interest
            roi = in_img_array[i:i+h.shape[0], j:j+h.shape[1]]
            # Multiply element-wise and sum
            out_img[i, j] = np.sum(roi * h)

    return out_img, out_origin
