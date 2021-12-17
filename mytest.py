import numpy

from camera.utils import create_cameras
import numpy as np
import json
import cv2

if __name__ == '__main__':

    img = cv2.imread('f_site/xr1.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (608, 608))

    area_a = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=numpy.int32)
    area_b = np.array([[20, 0], [40, 0], [40, 20], [20, 20]])
    print(area_b.dtype)
    mask = np.zeros([608, 608], dtype=np.uint8)

    cv2.polylines(img, [area_a], True [255])








