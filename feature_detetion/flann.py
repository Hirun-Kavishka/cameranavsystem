import cv2
import numpy as np

# Load the reference image 
reference_image = cv2.imread('images/loc00.jpg')

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Define the conversion factor from pixels to meters
pixels_to_meters = 0.1

# Initialize an array to store the relative positions
relative_positions = []

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the reference image
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_gray, None)

# FLANN parameters
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)

# FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Loop through the remaining images
for i in range(1, 5):
    # Load the current image
    image_path = 'images/loc{:02d}.jpg'.format(i)
    current_image = cv2.imread(image_path)

    # Convert the current image to grayscale
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current image
    keypoints_current, descriptors_current = sift.detectAndCompute(current_gray, None)

    # Match the descriptors using FLANN
    matches = flann.knnMatch(descriptors_ref, descriptors_current, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get the coordinates of the matched keypoints in both images
    ref_points = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    current_points = np.float32([keypoints_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate the homography matrix using RANSAC
    M, mask = cv2.findHomography(ref_points, current_points, cv2.RANSAC, 5.0)

    # Compute the relative position of the current image to the reference image in meters
    relative_x = M[0, 2] * pixels_to_meters
    relative_y = M[1, 2] * pixels_to_meters

    # Store the relative position in the array
    relative_positions.append((relative_x, relative_y))

    # Draw the matches on the current image
    matched_image = cv2.drawMatches(reference_gray, keypoints_ref, current_gray, keypoints_current, good_matches, None)

    # Display the matched image
    cv2.imshow('Matched Image {}'.format(i), matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Print the relative positions in meters
for i, (x, y) in enumerate(relative_positions):
    print('Relative position of loc{:02d}.jpg: x = {:.2f} meters, y = {:.2f} meters'.format(i + 1, x, y))
