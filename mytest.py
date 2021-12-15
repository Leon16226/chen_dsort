from camera.utils import create_cameras
import numpy as np
import json
import cv2

if __name__ == '__main__':


    img = cv2.imread('f_site/xr1.png', cv2.IMREAD_GRAYSCALE)
    # print(cv2.resize(a, (2, 2)))
    a = cv2.resize(img, (5, 5))
    b = cv2.resize(a, (3, 3), interpolation=cv2.INTER_LINEAR)
    print(a)
    print(b)
