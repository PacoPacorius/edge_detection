import numpy as np

def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, 
               V_min: int) -> np.ndarray, np.ndarray:
    "circ_hough will find a circle in a binary image. Accept input binary image 
    with only pixel values either 0 or 1. This binary image will be generated 
    with the use of log_edge or sobel_edge functions. R_max is the maximum radius of the 
    circle. dim is an array of 3 values, 2 of them divide the image the 
    horizontally and vertically, the third value divides the (0, R_max) 
    interval. dim is essentially the dimension of the matrix in hough's 
    algorithm. V_min is the minimum number of votes a hough cell must
    receive to be considered part of the circle. Output is the array centers, 
    the coordinates of the centers of the circles detected and the array radii,
    the radii of the circles detected."
                    

    """
    Circle detection using Hough transform.
    
    Parameters:
    -----------
    in_img_array : np.ndarray
        Binary input image with edges detected (values 0 or 1)
    R_max : float
        Maximum radius to search for
    dim : np.ndarray
        Array of 3 values: [dim_x, dim_y, dim_r]
        Dimensions for the accumulator array
    V_min : int
        Minimum number of votes for a circle to be detected
        
    Returns:
    --------
    centers : np.ndarray
        Array of detected circle centers (x, y)
    radii : np.ndarray
        Array of detected circle radii
    """
    # Extract dimensions for the Hough space
    dim_x, dim_y, dim_r = dim
    
    # Calculate step sizes
    step_x = in_img_array.shape[1] / dim_x
    step_y = in_img_array.shape[0] / dim_y
    step_r = R_max / dim_r
    
    # Create accumulator array (Hough space)
    accumulator = np.zeros((dim_y, dim_x, dim_r), dtype=int)
    
    # Get edge pixel coordinates
    edge_y, edge_x = np.where(in_img_array == 1)
    
    # Vote in the Hough space
    for i in range(len(edge_y)):
        y = edge_y[i]
        x = edge_x[i]
        
        # For each possible radius
        for r_idx in range(dim_r):
            r = (r_idx + 0.5) * step_r  # Use the middle of each radius bin
            
            # Vote for all possible center points at distance r from (x, y)
            for theta in np.linspace(0, 2*np.pi, 36):  # Sample 36 points around the circle
                # Calculate potential center
                a = int(x - r * np.cos(theta))
                b = int(y - r * np.sin(theta))
                
                # Check if the potential center is within image boundaries
                if 0 <= a < in_img_array.shape[1] and 0 <= b < in_img_array.shape[0]:
                    # Convert to accumulator indices
                    a_idx = int(a / step_x)
                    b_idx = int(b / step_y)
                    
                    # Ensure indices are within bounds
                    if 0 <= a_idx < dim_x and 0 <= b_idx < dim_y:
                        accumulator[b_idx, a_idx, r_idx] += 1
    
    # Find circles with votes ≥ V_min
    centers_y, centers_x, radii_idx = np.where(accumulator >= V_min)
    
    # If no circles found, return empty arrays
    if len(centers_y) == 0:
        return np.array([]), np.array([])
    
    # Convert accumulator indices to image coordinates and radii
    centers_x = (centers_x + 0.5) * step_x
    centers_y = (centers_y + 0.5) * step_y
    radii = (radii_idx + 0.5) * step_r
    
    # Combine centers into (x, y) coordinate pairs
    centers = np.column_stack((centers_x, centers_y))
    
    return centers, radii

