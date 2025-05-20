import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#from PIL import Image
import cv2 as cv
from sobel_edge import sobel_edge
from log_edge import log_edge
import circ_hough

""" 
In this file we call the necessary functions are called to detect the circle inside 
the sample image basketball_large.png.
We call the sobel_edge function for different values of thresh and display
all the resulting binary images. We plot a the number of points detected over 
the different values of thresh.

We display the resulting image from the log_edge function and compare with the 
results from the sobel_edge function.

We display the original image with the circle that circ_hough detected drawn on
top. We do this five times for different V_min values. We use ready Python 
functions to draw the circle.
"""

# Load the image
img1 = cv.imread('basketball_large.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img1, None, None, 0.1, 0.1, cv.INTER_AREA)
print('shape of original img =', img1.shape, '\t shape of shrunken img =', img.shape)


# Display the original image
plt.figure(figsize=(6, 5))
plt.subplot(211)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(212)
plt.imshow(img, cmap='gray')
plt.title('Shrunken Image')
plt.axis('off')
plt.show()

# Test Sobel edge detection with different thresholds
thresholds = [50, 100, 150, 200, 250]
edge_counts = []

plt.figure(figsize=(10, 10))
for i, thresh in enumerate(thresholds):
    print("Applying sobel operator edge detection ", i, " of ", len(thresholds), 
          "with thresh = ", thresh)

    # Apply Sobel edge detection
    edge_img = sobel_edge(img, thresh)

    # Count edge pixels
    edge_count = np.sum(edge_img)
    edge_counts.append(edge_count)

    # Display Sobel edge detection results
    plt.subplot(3, 3, i+2)
    plt.imshow(edge_img, cmap='gray')
    plt.title(f'Sobel, thresh={thresh}')
    plt.axis('off')

# Plot the number of edge points vs threshold
plt.subplot(331)
plt.plot(thresholds, edge_counts, 'b-o')
plt.xlabel('Threshold')
plt.ylabel('Number of Edge Points')
plt.title('Edge Points over Thresholds')
plt.grid(True)

# Apply LoG edge detection
print("Applying log operator edge detection")
log_edge_img = log_edge(img)

# Display LoG edge detection result
plt.subplot(337)
plt.imshow(log_edge_img, cmap='gray')
plt.title(f'LoG (edges={np.sum(log_edge_img)}), central differences')
plt.axis('off')
plt.show()

# Select one of the Sobel edge images for circle detection
# Using moderate threshold (150) for good balance
edge_img_for_hough = sobel_edge(img, 150)

# Set parameters for circle detection
R_max = min(img.shape) // 2  # Maximum radius is half of the smallest dimension
dim = np.array([32, 32, 32])  # Hough space dimensions

# Apply circle detection with different V_min values
#V_min_values = [3*10**3, 5*10**3, 1*10**4] #for 0.5 scale
V_min_values = [500, 750, 1000] #for 0.1 scale
#V_min_values = [5000]

# Create a new figure for circle detection results
plt.figure(figsize=(10, 6))

for i, V_min in enumerate(V_min_values):
    print("Sobel method finding circle with hough algorithm ", i, " of ", len(V_min_values), 
          " with V_min = ", V_min)
    centers, radii = circ_hough.circ_hough(edge_img_for_hough, R_max, dim, V_min)

    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Sobel Circle Detection (V_min={V_min})')

    # Draw detected circles
    ax = plt.gca()
    for center, radius in zip(centers, radii):
        circle = Circle([center[0], center[1]], radius, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)

    plt.axis('off')

    # Print information about detected circles
    print("V_min =", V_min, "\tDetected ", len(radii), " circles")

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))

V_min_values = [600, 800, 1000] #for 0.1 scale
for i, V_min in enumerate(V_min_values):
    print("LoG method finding circle with hough algorithm ", i, " of ", len(V_min_values), 
          " with V_min = ", V_min)
    centers, radii = circ_hough.circ_hough(log_edge_img, R_max, dim, V_min)

    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'LoG Circle Detection (V_min={V_min})')

    # Draw detected circles
    ax = plt.gca()
    for center, radius in zip(centers, radii):
        circle = Circle([center[0], center[1]], radius, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)

    plt.axis('off')

    # Print information about detected circles
    print("V_min =", V_min, "\tDetected ", len(radii), " circles")

plt.tight_layout()
plt.show()
