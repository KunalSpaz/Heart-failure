import cv2
import numpy as np

image = cv2.imread(r"C:\Users\student\Documents\220962316\OIP.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)

threshold = 0.01 * dst.max()
image[dst > threshold] = [0, 0, 255]  # Mark corners in red

cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
