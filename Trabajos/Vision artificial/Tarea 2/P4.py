import cv2
import numpy as np

img = cv2.imread("imagen1.jpg",cv2.IMREAD_GRAYSCALE)

h, w = img.shape[:2]
M = h * w


    