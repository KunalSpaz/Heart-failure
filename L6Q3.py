import cv2

cap = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))

    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8))

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('HoG Feature Descriptor - Webcam Feed', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
