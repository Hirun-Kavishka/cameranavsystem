import cv2
import numpy as np
import math

original_image = cv2.imread("original.jpg")
cutout_image = cv2.imread("cut6.png")

original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cutout_gray = cv2.cvtColor(cutout_image, cv2.COLOR_BGR2GRAY)

cutout_height, cutout_width = cutout_gray.shape

result = cv2.matchTemplate(original_gray, cutout_gray, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + cutout_width, top_left[1] + cutout_height)

original_with_mark = original_image.copy()
cv2.rectangle(original_with_mark, top_left, bottom_right, (0, 255, 0), 2)  # Draw a green rectangle around the cut-out

cutout_mask = np.zeros_like(original_image)
cutout_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (255, 0, 255)  # Create a pink mask for the cut-out

marked_image = cv2.addWeighted(original_with_mark, 0.7, cutout_mask, 0.3, 0)  # Combine the marked original and cut-out mask

# Calculate the displacement in x and y coordinates with exchanged polarity
original_middle_x = original_image.shape[1] // 2
original_middle_y = original_image.shape[0] // 2

cutout_middle_x = cutout_image.shape[1] // 2
cutout_middle_y = cutout_image.shape[0] // 2

displacement_x = top_left[0] - (original_middle_x - cutout_middle_x)  # Exchanged polarity for x-axis
displacement_y = top_left[1] - (original_middle_y - cutout_middle_y)  # Exchanged polarity for y-axis

print("Cut-out piece displacement (x, y):", displacement_x, displacement_y)

# Calculate the angle between the positive x-axis and the line connecting the two origins
angle = math.degrees(math.atan2(displacement_y, displacement_x))
angle = angle if angle >= 0 else 360 + angle  # Ensure the angle is positive

# Mark the origins and draw dotted lines for displacements with exchanged polarity
cv2.drawMarker(marked_image, (original_middle_x, original_middle_y), (0, 0, 255), cv2.MARKER_CROSS, 10, thickness=2)
cv2.drawMarker(marked_image, (cutout_middle_x + top_left[0], cutout_middle_y + top_left[1]), (255, 0, 0), cv2.MARKER_CROSS, 10, thickness=2)
cv2.line(marked_image, (original_middle_x, original_middle_y), (original_middle_x + displacement_x, original_middle_y + displacement_y), (0, 0, 255), 1, cv2.LINE_AA)
cv2.line(marked_image, (original_middle_x, original_middle_y), (original_middle_x + displacement_x, original_middle_y), (0, 255, 0), 1, cv2.LINE_AA)
cv2.line(marked_image, (original_middle_x, original_middle_y), (original_middle_x, original_middle_y + displacement_y), (255, 0, 0), 1, cv2.LINE_AA)

# Mark the angle value
cv2.putText(marked_image, "Angle: {:.2f}".format(360-angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Resize the marked image to 800x600
resized_image = cv2.resize(marked_image, (800, 600))

# Save the marked image as output.jpg
cv2.imwrite("output.jpg", resized_image)

cv2.imshow("Marked Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
