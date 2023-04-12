import cv2
import numpy as np

# Load image
img = cv2.imread('malarai1t.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Remove small objects
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) < 1000:
        cv2.drawContours(opening, [contour], 0, 0, -1)

# Invert binary image
mask = cv2.bitwise_not(opening)

# Apply mask to original image
result = cv2.bitwise_and(img, img, mask=mask)

# Find bounding box of darker regions
x, y, w, h = cv2.boundingRect(opening)

# Crop out darker regions
cropped = result[y:y+h, x:x+w]

# Display results
cv2.imshow('Original Image', img)
cv2.imshow('Processed Image', result)
cv2.imshow('Cropped Image', cropped)
print(type(cropped))
# Filename
filename = 'savedImage.jpg'
  
# Using cv2.imwrite() method
# Saving the image
# cv2.imwrite(filename, cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
