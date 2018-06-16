from detect import Detector
import cv2


detector = Detector(resolution=320)
loop = True

detect_path = 'imgs/img1.jpg'

path, objects, image = detector.detect_objects(detect_path)
cv2.imwrite('detected_image.jpg', image)
image_np = cv2.imread('detected_image.jpg')
cv2.imshow('images', image_np)
cv2.waitKey()
