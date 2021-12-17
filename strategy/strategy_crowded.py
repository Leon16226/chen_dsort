import numpy as np
from net.push import push
from .base_strategy import Strategy
import cv2
import copy
from shapely.geometry import Polygon

class CrowedStrategy(Strategy):
    # init
    def __init__(self, url, point, boxes, pool, im0s, labels, height, width, matrix_crowd, area_of_crowd):
        Strategy.__init__(self, url, point, boxes, pool, im0s, labels, height, width)
        self.matrix_crowed = matrix_crowd
        self.area_of_crowd = area_of_crowd

    def do(self, ):
        print("############################Crowed start##################################")
        self.lock.acquire()

        # counter
        key_counter = "counter"
        if key_counter in self.pool.keys():
            self.pool[key_counter] += 1

            if self.pool[key_counter] == 31:
                self.pool[key_counter] = 1
                self.matrix_crowed[:, :, 1] = self.matrix_crowed[:, :, 0]
                self.matrix_crowed[:, :, 0] = 0
        else:
            self.pool[key_counter] = 1

        # iter
        for box in self.boxes:
            x1, x2 = int(box[0] * 608), int(box[2] * 608)
            y1, y2 = int(box[1] * 608), int(box[3] * 608)

            # update
            self.matrix_crowed[y1:y2, x1:x2, 0] += 1

        # judge
        for i, area in enumerate(self.area_of_crowd):
            ratio_w = self.width / 608
            ratio_h = self.height / 608
            c_area = copy.deepcopy(area)
            c_area[:, 0] = (c_area[:, 0] / ratio_w).astype(np.int32)
            c_area[:, 1] = (c_area[:, 1] / ratio_h).astype(np.int32)

            # mask
            mask = np.zeros([608, 608], dtype=np.uint8)
            cv2.fillPoly(mask, [c_area], [255])
            img_now = self.matrix_crowed[:, :, 0]
            img_last = self.matrix_crowed[:, :, 1]
            roi_now = cv2.bitwise_and(img_now, img_now, mask=mask)
            roi_last = cv2.bitwise_and(img_last, img_last, mask=mask)

            # 面积
            my_area = Polygon(c_area).area
            mean_now = 1.0 * np.sum(roi_now) / (my_area * 30)
            mean_last = 1.0 * np.sum(roi_last) / (my_area * 30)
            weight = 1.0 * self.pool[key_counter] / 30
            mean_weight = (1 - weight) * mean_last + weight * mean_now

            my_key = str(c_area[0][0]) + str(c_area[0][1]) + str(c_area[2][0]) + str(c_area[2][1])

            # bayesian
            my_key_bayesian = my_key + 'bayesian'
            if my_key_bayesian in self.pool.keys():
                prior = self.pool[my_key_bayesian]
                prior = prior if prior > 0.01 else 0.01

                likelihood_a = np.power(mean_weight, 0.9)
                likelihood_b = -mean_weight + 1

                p1 = likelihood_a * prior
                p2 = likelihood_b * (1 - prior)

                uncertainty = np.mean(self.boxes[:, 5])

                poster = p1 / (p1 + p2)
                poster = poster if poster < 0.95 else 0.95

                self.pool[my_key_bayesian] = poster
            else:
                poster = 0.0
                self.pool[my_key_bayesian] = 0.01

            print('the rate of crowed', i, " ", poster)

            if poster < 0.10 and my_key in self.pool.keys():
                self.pool.pop(my_key)

            if poster > 0.70 and not (my_key in self.pool.keys()):
                my_im0s = copy.deepcopy(self.im0s)
                cv2.polylines(my_im0s, [area], True, (51, 51, 204), thickness=3)
                push(self.url, my_im0s, self.point, "crowed")

                self.pool[my_key] = False

        self.lock.release()
        print("#############################Crowed end################################")