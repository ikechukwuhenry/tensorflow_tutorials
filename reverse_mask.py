import cv2
import numpy as np

# Load binary mask image
mask = cv2.imread('savedImage.jpg', cv2.IMREAD_GRAYSCALE)

# Create a color map using numpy arrays
color_map = np.zeros((256, 1, 3), dtype=np.uint8)
color_map[:, 0, 2] = np.arange(256)  # Set red channel to a range of values
color_map[:, 0, 1] = 255 - np.arange(256)  # Set green channel to the complement of the red channel
color_map[:, 0, 0] = 255 - np.arange(256)  # Set blue channel to the complement of the red channel

# Apply the color map to the binary mask image
colored_mask = cv2.applyColorMap(mask, color_map)

# Display the results
cv2.imshow('Binary Mask', mask)
cv2.imshow('Colored Mask', colored_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
