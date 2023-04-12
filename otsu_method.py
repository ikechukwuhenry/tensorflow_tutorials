import cv2
import numpy as np

# Load image
img = cv2.imread('malarai1t.jpeg', 0)

# Apply Otsu thresholding
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Display results
cv2.imshow('Original Image', img)
cv2.imshow('Otsu Thresholding', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
