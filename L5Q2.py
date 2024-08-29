import cv2

def nothing(x):
    pass

image = cv2.imread(r"C:\Users\student\Downloads\VK.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image.")
    exit()

cv2.namedWindow('Binary Image')

cv2.createTrackbar('Threshold', 'Binary Image', 0, 255, nothing)

cv2.setTrackbarPos('Threshold', 'Binary Image', 127)

while True:

    threshold_value = cv2.getTrackbarPos('Threshold', 'Binary Image')

    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binary Image', binary_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
