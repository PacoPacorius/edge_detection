import numpy as np
import math

def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, 
               V_min: int) -> tuple:
    # divide image and radii in dim parts. new votes array.

    # r_step = R_max / dim[2]
    # a_step = img.shape[0] / dim[0]
    # b_step = img.shape[1] / dim[1]
    # vote = np.array((a_step, b_step, r_step))
    #
    # # kane ta r, a, b na einai sto meso tou kathe diasthmatos, twra einai sthn arxh
    # for each pixel with a value of 1 in the input img with coords (x,y)
    #   for each radius r, where r in [0, R_max] with step r_step
    #       for each a in [0, img.shape[0]] with step a_step
    #           for each b in [0, img.shape[0]] with step b_step
    #              if (a - x)**2 + (b - y)**2 - r**2 < 1e-6
    #              then add a vote to vote(a'/ N, b' / N, r / N)
    # a, b, r = a_step * index(max(vote)), b_step * index(max(vote)), r_step * index(max(vote))
    # return tripleta a, b, r me ta perissotera votes

    r_step = int(R_max / dim[2])
    cols_step = int(in_img_array.shape[1] / dim[0])
    rows_step = int(in_img_array.shape[0] / dim[1])
    vote = np.zeros((dim[0], dim[1], dim[2]))


    for columns in range(0, in_img_array.shape[1]):
        for rows in range(0, in_img_array.shape[0]):
            if in_img_array[rows,columns] == 1:
                for r_idx in range(0, dim[2]):
                    r = (r_idx + 0.5) * r_step
                    for a_idx in range(0, dim[0]):
                        a = (a_idx + 0.5) * cols_step
                        for b_idx in range(0, dim[1]):
                            b = (b_idx + 0.5) * rows_step
                            tolerance = max(1e-2, 0.1 * r)
                            dist = np.hypot(a - columns, b - rows)
                            if abs(dist - r) < tolerance:
                                vote[a_idx, b_idx, r_idx] = vote[a_idx, b_idx, r_idx] + 1
    # Find the bin with maximum votes
    max_votes = np.max(vote)

    # Check if max votes exceeds the threshold
    if max_votes < V_min:
        # Return empty arrays if no circle meets the threshold
        return np.array([]), np.array([])

    # Get indices of maximum vote
    max_indices = np.unravel_index(np.argmax(vote), vote.shape)
    a_idx, b_idx, r_idx = max_indices

    # Calculate center coordinates and radius (center of the bins)
    center_x = (a_idx + 0.5) * cols_step
    center_y = (b_idx + 0.5) * rows_step
    radius = (r_idx + 0.5) * r_step

    # Create return arrays
    centers = np.array([[center_x, center_y]])
    radii = np.array([radius])

    print(f"Circle detected: center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}, votes={max_votes}")

    return centers, radii
##input()
    #max_indices = np.zeros(3)
    #max_indices = np.argmax(vote, axis = 0)
    #centers = np.zeros(2)
    #centers = np.array([max_indices[0] * a_step, max_indices[1] * b_step])
    #radius = np.array([max_indices[2] * r_step])
    #return centers, radius

def circ_hough3(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int) -> tuple:
    height, width = in_img_array.shape
    accumulator = np.zeros((dim[0], dim[1], dim[2]), dtype=int)

    a_step = width / dim[0]
    b_step = height / dim[1]
    r_step = R_max / dim[2]

    edge_points = np.argwhere(in_img_array == 1)

    for y, x in edge_points:
        for r_idx in range(dim[2]):
            r = (r_idx + 0.5) * r_step
            for angle in np.linspace(0, 2*np.pi, 72):  # finer angular resolution
                a = x - r * np.cos(angle)
                b = y - r * np.sin(angle)

                a_idx = int(a / a_step)
                b_idx = int(b / b_step)

                if 0 <= a_idx < dim[0] and 0 <= b_idx < dim[1]:
                    accumulator[a_idx, b_idx, r_idx] += 1

    max_votes = np.max(accumulator)
    if max_votes < V_min:
        return np.array([]), np.array([])

    a_idx, b_idx, r_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    center_x = (a_idx + 0.5) * a_step
    center_y = (b_idx + 0.5) * b_step
    radius = (r_idx + 0.5) * r_step

    return np.array([[center_x, center_y]]), np.array([radius])

def circ_hough2(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int) -> tuple:
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
    # Get image dimensions
    height, width = in_img_array.shape

    # Create accumulator array - using the dimensions specified
    accumulator = np.zeros((dim[0], dim[1], dim[2]), dtype=int)

    # Calculate step sizes
    a_step = width / dim[0]  # x-direction step size
    b_step = height / dim[1]  # y-direction step size
    r_step = R_max / dim[2]   # radius step size

    # Find all edge points (where pixel value is 1)
    edge_points = np.where(in_img_array == 1)
    y_points = edge_points[0]
    x_points = edge_points[1]

    # More efficient circle detection
    # For each edge point, vote for possible circle centers
    for i in range(len(y_points)):
        y, x = y_points[i], x_points[i]

        # For each possible radius
        for r_idx in range(dim[2]):
            radius = (r_idx + 0.5) * r_step  # Use middle of the bin

            # Instead of checking every possible (a,b) pair,
            # we'll directly compute votes for points on circles
            for angle in np.linspace(0, 2*np.pi, 36):  # 36 points around the circle
                # Calculate potential center
                a = x - radius * np.cos(angle)
                b = y - radius * np.sin(angle)

                # Get indices in accumulator array
                a_idx = int(a / a_step)
                b_idx = int(b / b_step)

                # Check if indices are within bounds
                if 0 <= a_idx < dim[0] and 0 <= b_idx < dim[1]:
                    accumulator[a_idx, b_idx, r_idx] += 1

    # Find circles with votes >= V_min
    centers = []
    radii = []

    # Find the bin with the maximum votes
    max_indices = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    a_idx, b_idx, r_idx = max_indices

    # Convert to image coordinates
    center_x = (a_idx + 0.5) * a_step
    center_y = (b_idx + 0.5) * b_step
    radius = (r_idx + 0.5) * r_step

    centers.append([center_x, center_y])
    radii.append(radius)


    return np.array(centers), np.array(radii)
