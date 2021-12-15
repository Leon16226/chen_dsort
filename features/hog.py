import cv2
import numpy as np
import math

# 方向直方图

class HOG(object):

    def __init__(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        self.cell_size = 16
        self.bin_size = 9
        self.unit = 360 // self.bin_size
        self.height, self.width = img.shape

        # cell矩阵
        self.cell_gradient_vector = np.zeros((self.height // self.cell_size, self.width // self.cell_size, self.bin_size))

    """
    sobel算子计算水平和垂直方向的差分
    """
    def compute_image_gradient(self,):
        # 边缘检测算子
        x_values = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        y_values = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = abs(cv2.addWeighted(x_values, 0.5, y_values, 0.5, 0))
        angle = cv2.phase(x_values, y_values, angleInDegrees=True)  # 返回角度

        return magnitude, angle

    def choose_bins(self, gradient_angle):
        idx = int(gradient_angle / self.unit)
        mod = gradient_angle % self.unit
        return  idx, (idx+1) % self.bin_size, mod

    """
    计算细胞单元内的梯度方向
    """
    def compute_cell_gradient(self, cell_magnitude, cell_angle):
        centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.choose_bins(gradient_angle)
                # 根据角度将近程度，分别对相邻的两个区间进行加权
                centers[min_angle] += (strength * (1 - (mod / self.unit)))
                centers[max_angle] += (strength * (mod / self.unit))
        return  centers

    def normalized(self, cell_gradient_vector):
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude : [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector ,magnitude)
                hog_vector.append(block_vector)
        return hog_vector

    def calculate(self):
        magnitude, angle = self.compute_image_gradient(self.img)

        for i in range(self.cell_gradient_vector.shape[0]):
            for j in range(self.cell_gradient_vector.shape[1]):
                cell_magnitude = magnitude[i * self.cell_size:(i+1) * self.cell_size,
                                 j * self.cell_size:(j+1) * self.cell_size]
                cell_angle = magnitude[i * self.cell_size:(i+1) * self.cell_size,
                                 j * self.cell_size:(j+1) * self.cell_size]
                self.cell_gradient_vector[i][j] = self.compute_cell_gradient(cell_magnitude, cell_angle,
                                                                             self.bin_size, self.unit)
        hog_vector = self.normalized(self.cell_gradient_vector)
        return  hog_vector

    def calculate_from_opencv(self):
        hog = cv2.HOGDescriptor()
        des = hog.compute(self.img)
        return des






