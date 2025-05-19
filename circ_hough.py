import numpy as np
import math

def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, 
               V_min: int) -> tuple:
    """
    Detects circles in a binary image using the Hough transform.

    Args:
        in_img_array (np.ndarray): Binary input image (0 or 1)
        R_max (float): Maximum radius of the circle to detect
        dim (np.ndarray): Dimensions for binning [dim_x, dim_y, dim_r]
        V_min (int): Minimum number of votes required to consider a circle

    Returns:
        np.ndarray: Array of coordinates of circle centers
        np.ndarray: Array of radii of detected circles
    """

    # Calculate steps and initialize hough voting matrix
    r_step = R_max / dim[2]
    cols_step = in_img_array.shape[1] / dim[0]
    rows_step = in_img_array.shape[0] / dim[1]
    vote = np.zeros((dim[0], dim[1], dim[2]))


    for columns in range(0, in_img_array.shape[1]):
        for rows in range(0, in_img_array.shape[0]):
            # For each edge pixel
            if in_img_array[rows,columns] == 1:
                # For each radius
                for r_idx in range(0, dim[2]):
                    r = (r_idx + 0.5) * r_step
                    for a_idx in range(0, dim[0]):
                        a = (a_idx + 0.5) * cols_step
                        for b_idx in range(0, dim[1]):
                            b = (b_idx + 0.5) * rows_step
                            # if the edge pixel belongs to the circle with 
                            # center (a,b) and radius r, add a vote to the Hough
                            # voting matrix
                            tolerance = max(1e-2, 0.1 * r)
                            dist = np.hypot(a - columns, b - rows)
                            if abs(dist - r) < tolerance:
                                vote[a_idx, b_idx, r_idx] = vote[a_idx, b_idx, r_idx] + 1


 # Find circles with votes >= V_min
    centers = []
    radii = []
    
    for a_idx in range(dim[0]):
        for b_idx in range(dim[1]):
            for r_idx in range(dim[2]):
                if vote[a_idx, b_idx, r_idx] >= V_min:
                    # Convert back to image coordinates (center of the bin)
                    center_x = (a_idx + 0.5) * cols_step
                    center_y = (b_idx + 0.5) * rows_step
                    radius = (r_idx + 0.5) * r_step
                    
                    centers.append([center_x, center_y])
                    radii.append(radius)
    
    # If no circles found with votes >= V_min, return the best match
    if len(centers) == 0 and np.max(vote) > 0:
        ### Uncomment this code to return the most voted circle if V_min is too high ###
#        # Find the bin with the maximum votes
#        max_indices = np.unravel_index(np.argmax(vote), vote.shape)
#        a_idx, b_idx, r_idx = max_indices
#        
#        # Convert to image coordinates
#        center_x = (a_idx + 0.5) * cols_step
#        center_y = (b_idx + 0.5) * rows_step
#        radius = (r_idx + 0.5) * r_step
#        
#        centers.append([center_x, center_y])
#        radii.append(radius)
        
        print(f"No circles found with votes >= {V_min}.")
        #print("Returning best match with {vote[a_idx, b_idx, r_idx]} votes.")
    
    return np.array(centers), np.array(radii)
