import cv2
import numpy
import numpy as np


# 计算均值hash数值
def sim_hash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    avr = np.mean(gray)
    my_hash = []
    for v in np.nditer(gray):
        if v > avr:
            my_hash.append(1)
        else:
            my_hash.append(0)

    return np.array(my_hash, dtype=numpy.int8)


# 计算差分hash数值
def sim_dhash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    my_hash = []
    for x in range(8):
        for y in range(8):
            if gray[x][y] > gray[x][y+1]:
                my_hash.append(1)
            else:
                my_hash.append(0)
    return np.array(my_hash, dtype=numpy.int8)


# 计算汉明距离
def calculate_hanming(hash1, hash2):
    if hash2 is None:
        hash2 = np.zeros(64, dtype=numpy.int8)

    num = 0
    for i, v in enumerate(np.nditer(hash2)):
        if hash1[i] != v:
            num += 1

    return num


# 颜色直方图
def sim_color_histogram(img):
    b, g, r = cv2.split(img)

    histImgB = cv2.calcHist([b], [0], None, [256], [0, 255])
    histImgG = cv2.calcHist([g], [0], None, [256], [0, 255])
    histImgR = cv2.calcHist([r], [0], None, [256], [0, 255])

    hist = (histImgB + histImgG + histImgR) / 3

    return hist


def calculate_color(hist1, hist2):

    if hist2 is None:
        hist2 = np.zeros((256, 1), dtype=numpy.int8)

    similarity = cv2.compareHist(hist1, hist2, 0)

    return similarity



