import cv2
import numpy as np

# Load the image
img = cv2.imread('test1.png')

# Display the original image
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Convert the image to HSL color space
hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# Display the HSL image
cv2.imshow('HSL Image', hsl)
cv2.waitKey(0)

# Define the threshold range for the hue channel
hue_min = 0
hue_max = 360

# Threshold the hue channel to get a binary mask for the colors of the shapes
hue_mask = cv2.inRange(hsl[:,:,0], hue_min, hue_max)

# Display the hue mask
cv2.imshow('Hue Mask', hue_mask)
cv2.waitKey(0)

# Find the contours for the shapes
contours, hierarchy = cv2.findContours(hue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
