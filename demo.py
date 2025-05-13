import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough

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
img = np.array(Image.open('basketball_large.png').convert('L'))

# Display the original image
plt.figure(figsize=(12, 10))
plt.subplot(331)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Test Sobel edge detection with different thresholds
thresholds = [50, 100, 150, 200, 250]
edge_counts = []

for i, thresh in enumerate(thresholds):
    # Apply Sobel edge detection
    edge_img = sobel_edge(img, thresh)

    # Count edge pixels
    edge_count = np.sum(edge_img)
    edge_counts.append(edge_count)

    # Display Sobel edge detection results
    plt.subplot(3, 3, i+2)
    plt.imshow(edge_img, cmap='gray')
    plt.title(f'Sobel (thresh={thresh}, edges={edge_count})')
    plt.axis('off')

# Plot the number of edge points vs threshold
plt.subplot(337)
plt.plot(thresholds, edge_counts, 'b-o')
plt.xlabel('Threshold')
plt.ylabel('Number of Edge Points')
plt.title('Edge Points vs Threshold')
plt.grid(True)

# Apply LoG edge detection
log_edge_img = log_edge(img)

# Display LoG edge detection result
plt.subplot(338)
plt.imshow(log_edge_img, cmap='gray')
plt.title(f'LoG (edges={np.sum(log_edge_img)})')
plt.axis('off')

# Select one of the Sobel edge images for circle detection
# Using moderate threshold (100) for good balance
edge_img_for_hough = sobel_edge(img, 100)

# Set parameters for circle detection
R_max = min(img.shape) // 2  # Maximum radius is half of the smallest dimension
dim = np.array([50, 50, 30])  # Hough space dimensions

# Apply circle detection with different V_min values
V_min_values = [50, 100, 150, 200, 250]

# Create a new figure for circle detection results
plt.figure(figsize=(15, 10))

for i, V_min in enumerate(V_min_values):
    centers, radii = circ_hough(edge_img_for_hough, R_max, dim, V_min)

    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Circle Detection (V_min={V_min})')

    # Draw detected circles
    ax = plt.gca()
    for center, radius in zip(centers, radii):
        circle = Circle(center, radius, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)

    plt.axis('off')

    # Print information about detected circles
    print(f"V_min = {V_min}, Detected {len(radii)} circles")
    for j, (center, radius) in enumerate(zip(centers, radii)):
        print(f"  Circle {j+1}: center = ({center[0]:.1f}, {center[1]:.1f}), radius = {radius:.1f}")

plt.tight_layout()
plt.show()

