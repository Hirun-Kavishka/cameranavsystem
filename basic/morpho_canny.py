import cv2
import numpy as np

# Load the image
img = cv2.imread('test2.png')

# Convert to grayscale
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
cv2.imshow('Morphological image', morph)
cv2.waitKey(0)



# Apply Canny edge detection
edges = cv2.Canny(morph, 100, 200)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Find contours and draw them on the original image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate areas and print them
for contour in contours:
    area = cv2.contourArea(contour)
    print("Area:", area)

cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# Display the final result
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
