import cv2
import numpy as np

# Load the image
img = cv2.imread('test2.png')

# Display the original image
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the binary image
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)

# Apply morphological operations to remove noise and fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Display the morphological image
cv2.imshow('Morphological Image', morph)
cv2.waitKey(0)

# Find the contours of the shapes
contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
for contour in contours:
    color = np.random.randint(0, 255, size=3).tolist()
    cv2.drawContours(img, [contour], -1, color, 2)

# Display the image with contours
cv2.imshow('Image with Contours', img)
cv2.waitKey(0)

# Calculate the area of the contours
areas = [cv2.contourArea(contour) for contour in contours]

# Display the areas
print('Areas:', areas)

# Close all windows
cv2.destroyAllWindows()
