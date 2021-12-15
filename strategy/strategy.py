import cv2
import numpy as np
import threading
import time
from threading import Thread
from deepsort.utils import iou, calc_iou, compute_color_for_id
from net.push import push
from .base_strategy import Strategy
from utils.shape import poly_area
from utils.norm import start_block
from utils.mystrategy import amend_sim
from deepsort.utils import calc_iou
from .img_similarity import calculate_hanming, sim_hash

thresholds = [20, 3, 3, 20, 0.8]

# 0.异常停车--------------------------------------------------------------------------------------------------------------
class IiiParkStrategy(Strategy):

    # init
    def __init__(self, url, point, boxes, pool, im0s, labels, height, width, matrix_park,):
        Strategy.__init__(self, url, point, boxes, pool, im0s, labels, height, width)
        self.matrix_park = matrix_park

    def do(self,):
        print("############################IllegalPark start##################################")
        self.lock.acquire()

        # cut down
        self.matrix_park[:, :, 0] = self.matrix_park[:, :, 0] * (2 / 3)

        updates = []
        for box in self.boxes:
            print('next box')

            # 这一帧
            xyxy1 = box[0:4]
            conf = box[5]
            cx1, cy1 = (int((box[0] + box[2]) / 2 * 608), int((box[1] + box[3]) / 2 * 608))  # 中心坐标
            h, w = box[3] - box[1], box[2] - box[0]  # 高、宽
            aspect_ratio1 = h / w  # 长宽比
            size1 = h * w  # 面积
            # 直方图
            img = self.im0s[int(box[1] * self.height):int(box[3] * self.height),
                            int(box[0] * self.width):int(box[2] * self.width)]
            h1 = cv2.calcHist([img], [1], None, [256], [0, 256])
            h1 = cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX, -1)

            # 上一帧
            aspect_ratio2 = self.matrix_park[cx1, cy1, 1]
            size2 = self.matrix_park[cx1, cy1, 2]
            xyxy2 = self.matrix_park[cx1, cy1, 3:7]
            # histogram
            my_key = str(cx1) + str(cy1)
            if my_key in self.pool.keys():
                h2 = self.pool[my_key]
                self.pool.pop(my_key)
            else:
                h2 = 1.0 - h1

            # 计算相似度
            sim1 = 1 - np.abs(aspect_ratio2 - aspect_ratio1) / aspect_ratio1
            sim2 = 1 - np.abs(size2 - size1) / size1
            sim3 = calc_iou(xyxy2, xyxy1)
            sim4 = cv2.compareHist(h1, h2, 0)

            # 修正
            sim1 = amend_sim(sim1)  # 长宽比
            sim2 = amend_sim(sim2)  # 面积
            sim3 = amend_sim(sim3)  # iou
            sim4 = amend_sim(sim4)  # 直方图
            print('sim1:', sim1)
            print('sim2:', sim2)
            print('sim3:', sim3)
            print('sim4:', sim4)

            # 加权平均 * 置信度
            sim = (sim1 + sim2 + sim3 + sim4) / 4 * conf
            print('sim', sim)

            # likelihood
            likelihood1 = np.power(sim, 1/1.2)
            likelihood2 = -sim + 1
            print('likelihood1', likelihood1)
            print('likelihood2', likelihood2)

            # inference
            prior = self.matrix_park[cx1, cy1, 0] * 1.5
            prior = prior if prior > 0.0001 else 0.0001
            poster = likelihood1 * prior / (likelihood1 * prior + likelihood2 * (1 - prior))
            poster = 0.95 if poster > 1 else poster

            print('the rate of illegalPark', poster)

            circles_lock_w = int(w * 608 * 0.25)
            circles_lock_h = int(h * 608 * 0.25)
            lock_area = self.matrix_park[cx1 - circles_lock_w:cx1 + circles_lock_w, cy1 - circles_lock_h:cy1 + circles_lock_h]
            judge = lock_area > 0.5
            # 解锁 : 加锁区域内，全部低于某个阈值再解锁
            if prior < 0.01 and not np.any(judge):
                print('解锁')
                self.matrix_park[cx1 - circles_lock_w:cx1 + circles_lock_w, cy1 - circles_lock_h:cy1 + circles_lock_h, 7] = 0

            # post
            if poster > 0.90 and self.matrix_park[cx1, cy1, 7] == 0:
                self.pbox = [box]
                self.draw()
                push(self.url, self.im0s, self.point, "illegalPark")

                # 加锁
                print('加锁')
                self.matrix_park[cx1 - circles_lock_w:cx1 + circles_lock_w, cy1 - circles_lock_h:cy1 + circles_lock_h, 7] = 1

            circles = 2
            # update
            boundary_x1 = cx1 - circles if cx1 - circles > 0 else 0
            boundary_x2 = cx1 + circles if cx1 + circles < 608 else 608
            boundary_y1 = cy1 - circles if cy1 - circles > 0 else 0
            boundary_y2 = cy1 + circles if cy1 + circles < 608 else 608

            self.matrix_park[boundary_x1:boundary_x2, boundary_y1:boundary_y2, 0] = poster
            self.matrix_park[boundary_x1:boundary_x2, boundary_y1:boundary_y2, 1] = aspect_ratio1
            self.matrix_park[boundary_x1:boundary_x2, boundary_y1:boundary_y2, 2] = size1
            self.matrix_park[boundary_x1:boundary_x2, boundary_y1:boundary_y2, 3:7] = xyxy1

            my_key = str(cx1) + str(cy1)
            self.pool[my_key] = h1.copy()

        self.lock.release()
        print("#############################IllegalPark end################################")


# 1. 行人检测------------------------------------------------------------------------------------------------------------
class PeopleStrategy(Strategy):

    def do(self,):
        # init
        peoples = self.boxes[0]
        cars = self.boxes[1]
        self.boxes = peoples

        if len(peoples) == 0:
            return None

        # 加一个空车
        if cars.size == 0:
            cars = np.array([[0, 0, 10, 10, -1, -1]])

        # iou
        ious = calc_iou(peoples[:, 0:4], cars[:, 0:4])
        print("people ious:", ious)

        for j, box in enumerate(self.boxes):
            # init
            bboxes = box[0:4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                o = iou(bboxes, p[1:5])
                if p[0] < self.threshold and o > 0.30:

                    pious = ious[j]
                    index = np.argmax(pious)

                    # 行人格外策略----------------------------------------------------------------------------------------
                    # 1. 人和车重叠iou > 0
                    if pious[index] > 0:
                        states = 0
                    else:
                        states = p[0] + 1 if o > 0.60 else p[0]
                    break
                    # 行人格外策略----------------------------------------------------------------------------------------
                elif p[0] >= self.threshold and o > 0.60:
                    states = self.threshold + 1
                    break

            self.pool.append([states, box[0], box[1], box[2], box[3]])
            self.lock.release()

            print("当前状态为：", states)

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "peopleOrNoVehicles")


# 2. 抛洒物--------------------------------------------------------------------------------------------------------------
class MaterialStrategy(Strategy):
    def do(self):
        for j, box in enumerate(self.boxes):
            # init
            bboxes = box[0:4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                o = iou(bboxes, p[1:5])
                if p[0] < self.threshold and o > 0.1:
                    print("抛撒物状态加1")
                    states = p[0] + 1
                    break
                elif p[0] >= self.threshold and o > 0.1:
                    states = self.threshold + 1
                    break

            self.pool.append([states, box[0], box[1], box[2], box[3]])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "throwThings")


# 3. 应急车道异常行驶-----------------------------------------------------------------------------------------------------
class illegalDriving(Strategy):

    def do(self,):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]
            color = compute_color_for_id(id)

            # lock
            self.lock.acquire()
            states = 0
            points = ''
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    states = p[1] + 1
                    points = p[2] + str(int((box[0] + box[2])/2)) + ',' + str(int((box[1] + box[3])/2)) + ','
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            print("id:", id)
            print("当前状态：", states)
            self.pool.append([box[4], states, points])  # ponits会被程序释放掉
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()

                # 画点
                points = points.split(',')
                mpoints = []
                for i, p in enumerate(points[0:-1]):
                    if i % 2 == 0:
                        mpoints.append((int(p), int(points[i + 1])))
                for point in mpoints:
                    print("画点")
                    cv2.circle(self.im0s, point, 5, color, -1)

                push(self.opt, self.im0s, "illegalDriving")


# 4. 缓行---------------------------------------------------------------------------------------------------------------
class crowedSrtategy(Strategy):
    def do(self, ):
        # 1. p(拥堵) = 0.10 p（不拥堵）= 0.90
        # 2. p(空间占有率x, 时间占有率y|拥堵) = x0.5次
        # 3. p(空间占有率x，时间占有率y|不拥堵) = -x + 1
        # 缺点：只要空间占有率一高就认为是拥堵了

        def f(x):
            return np.power(x, 0.25)

        def w(x):
            return -x + 1

        p_crow = 0.10
        p_no_crow = 0.90
        p_b = 0.0
        p_b_a = 0.0

        t_boxes = self.boxes[0]
        space_rate = self.boxes[1]
        t_pass = self.boxes[2]
        car_track_rate = self.boxes[3]
        crowed_block = self.boxes[4]

        my_counts = 60
        counts = 0
        t_counts = 0
        time_rate = 0.0
        p_rate = 0.0

        self.lock.acquire()
        # 时间占有率
        if len(self.pool) > 0:
            if self.pool[-1][1] > my_counts:
                counts = 0
                t_counts = 0
                time_rate = 0
                p_rate = self.pool[-1][4]  # 更新上一轮的时间占有率
            else:
                counts = self.pool[-1][1]
                t_counts = self.pool[-1][2] + 1 if t_pass else self.pool[-1][2]
                # 计算这一帧的时间占有率
                time_rate = t_counts / my_counts
                # 上一轮时间占有率
                p_rate = self.pool[-1][3]
                # 加权求和
                weight = counts / my_counts
                time_rate = (1 - weight) * p_rate + weight * time_rate
                time_rate = time_rate if time_rate > 0.10 else 0.10

            counts += 1

        print("%%%%%%%%%%%%%%%%%%% 时间占有率 %%%%%%%%%%%%%%%%%%%%%：", time_rate)
        print("时间通过数量：", t_counts)

        # Bayes
        if len(self.pool) > 0:
            p_crow = self.pool[-1][0]
            p_no_crow = 1 - p_crow

        all_event_rate = space_rate * time_rate * car_track_rate
        p_b = f(all_event_rate) * p_crow + w(all_event_rate) * p_no_crow
        p_b_a = f(all_event_rate)
        print("p(空间占有率x，时间占有率y)", p_b)
        print("p(空间占有率x，时间占有率|拥堵)", p_b_a)
        p_a_b = p_b_a * p_crow / p_b
        # [0拥堵概率，1计数器，2这一轮通过数量，3上一轮时间占有率，4这一轮时间占有率]
        p_a_b = p_b_a if p_a_b > 0.1 else 0.1
        self.pool.append([p_a_b, counts, t_counts, p_rate, time_rate])

        print("当前拥堵的概率为：", p_a_b)

        # post
        if p_a_b >= self.threshold and not crowed_block[0]:
            self.pbox = t_boxes
            self.draw()
            # push(self.opt, self.im0s, "crowed")
            print("push crowed yes！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            # block
            thread_ptz = Thread(target=start_block, args=(crowed_block,), daemon=True)
            thread_ptz.start()
        self.lock.release()






