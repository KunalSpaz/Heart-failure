import cv2

cap = cv2.VideoCapture(0)

orb = cv2.ORB_create()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 255, 0),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('ORB Feature Descriptor - Webcam Feed', frame_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
