import cv2
import numpy as np

# Load the reference
reference_image = cv2.imread('images/loc00.jpg')

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Define the conversion factor from pixels to meters
pixels_to_meters = 0.1

# Initialize an array to store the relative positions
relative_positions = []

# Create a SURF object
surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and compute descriptors for the reference image
keypoints_ref, descriptors_ref = surf.detectAndCompute(reference_gray, None)

# Loop through the remaining images
for i in range(1, 5):
    # Load the current image
    image_path = 'images/loc{:02d}.jpg'.format(i)
    current_image = cv2.imread(image_path)

    # Convert the current image to grayscale
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current image
    keypoints_current, descriptors_current = surf.detectAndCompute(current_gray, None)

    # Create a BFMatcher object
    matcher = cv2.BFMatcher()

    # Match the descriptors
    matches = matcher.match(descriptors_ref, descriptors_current)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the top match (best match)
    best_match = matches[0]

    # Get the coordinates of the matched keypoints in both images
    ref_point = keypoints_ref[best_match.queryIdx].pt
    current_point = keypoints_current[best_match.trainIdx].pt

    # Compute the relative position of the current image to the reference image in meters
    relative_x = (current_point[0] - ref_point[0]) * pixels_to_meters
    relative_y = (current_point[1] - ref_point[1]) * pixels_to_meters

    # Store the relative position in the array
    relative_positions.append((relative_x, relative_y))

    # Draw the match on the current image
    matched_image = cv2.drawMatches(reference_gray, keypoints_ref, current_gray, keypoints_current, [best_match], None)

    # Display the matched image
    cv2.imshow('Matched Image {}'.format(i), matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Print the relative positions in meters
for i, (x, y) in enumerate(relative_positions):
    print('Relative position of loc{:02d}.jpg: x = {:.2f} meters, y = {:.2f} meters'.format(i + 1, x, y))
