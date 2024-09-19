import cv2
import numpy as np

def gaussian_blur(image, sigma):

    ksize = int(6 * sigma + 1) | 1 
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def difference_of_gaussians(image, sigma1, sigma2):

    blurred1 = gaussian_blur(image, sigma1)
    blurred2 = gaussian_blur(image, sigma2)
    return cv2.subtract(blurred1, blurred2)

def detect_keypoints(image):

    harris_corners = cv2.cornerHarris(image, 2, 3, 0.04)
    keypoints = np.argwhere(harris_corners > 0.01 * harris_corners.max())
    return keypoints

def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dog_image = difference_of_gaussians(gray, 1.0, 1.6)

        keypoints = detect_keypoints(dog_image)

        for y, x in keypoints:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)

        cv2.imshow('Basic SIFT-like Feature Descriptor - Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
