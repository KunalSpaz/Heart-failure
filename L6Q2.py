import cv2

image = cv2.imread(r"C:\Users\student\Documents\220962316\OIP.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()

keypoints = fast.detect(gray, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (255, 0, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('FAST Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
