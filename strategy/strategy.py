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


# 0.异常停车---------------------------------------------------------------------------------------------------------------
class CarStrategy(Strategy):

    def do(self,):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]
            bboxes = box[0:4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    o = iou(bboxes, p[2:6])
                    # print("当前p时间：", p[6])
                    # print("iou:", o)
                    # print("thread id:", threading.currentThread().ident)
                    states = p[1] + 1 if o > 0.95 else p[1]
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            # print("id:", id)
            print("当前状态为：", states)

            self.pool.append([box[4], states, box[0], box[1], box[2], box[3], int(round(time.time() * 1000))])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "illegalPark")


# 1. 行人检测------------------------------------------------------------------------------------------------------------
class PeopleStrategy(Strategy):

    def do(self,):
        # init
        cars = self.boxes[self.boxes[:, 5] == 0]
        peoples = self.boxes[self.boxes[:, 5] == 8]
        self.boxes = peoples

        if self.boxes.size == 0:
            return None

        # 加一个空车
        if cars.size == 0:
            cars = np.array([[0, 0, 10, 10, -1, -1]])

        # iou
        ious = calc_iou(peoples[:, 0:4], cars[:, 0:4])
        print("people ious:", ious)

        for j, box in enumerate(self.boxes):
            # 参数初始化
            id = box[4]

            # lock
            self.lock.acquire()
            states = 0
            quadrant = -1
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:

                    pious = ious[j]
                    index = np.argmax(pious)
                    car = cars[index]

                    # 行人格外策略----------------------------------------------------------------------------------------
                    # 1. 人和车重叠iou > 0
                    if pious[index] > 0:
                        o = np.array([(car[0] + car[2])/2, (car[1] + car[3])/2])
                        x = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                        y = x - o

                        # quadrant
                        # (-1, -1)  (1, -1)
                        # (-1,  1)  (1,  1)
                        if y[0] < 0 and y[1] < 0:
                            quadrant = 0
                        elif y[0] < 0 and y[1] > 0:
                            quadrant = 1
                        elif y[0] > 0 and y[1] > 0:
                            quadrant = 2
                        elif y[0] > 0 and y[1] < 0:
                            quadrant = 3

                        if p[2] != quadrant and quadrant != -1:
                            states = p[1] + 1
                    else:
                        states = p[1] + 1
                    break
                    # 行人格外策略----------------------------------------------------------------------------------------
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            self.pool.append([box[4], states, quadrant])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "peopleOrNoVehicles")


# 2. 抛洒物--------------------------------------------------------------------------------------------------------------
class MaterialStrategy(Strategy):
    def do(self):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    print("抛撒物状态加1")
                    states = p[1] + 1
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break


            self.pool.append([box[4], states])
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

            # print("id:", id)
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

        my_counts = 10
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

        # print("%%%%%%%%%%%%%%%%%%% 时间占有率 %%%%%%%%%%%%%%%%%%%%%：", time_rate)
        # print("时间通过数量：", t_counts)

        # Bayes
        if len(self.pool) > 0:
            p_crow = self.pool[-1][0]
            p_no_crow = 1 - p_crow

        all_event_rate = space_rate * time_rate * car_track_rate
        p_b = f(all_event_rate) * p_crow + w(all_event_rate) * p_no_crow
        p_b_a = f(all_event_rate)
        # print("p(空间占有率x，时间占有率y)", p_b)
        # print("p(空间占有率x，时间占有率|拥堵)", p_b_a)
        p_a_b = p_b_a * p_crow / p_b
        # [0拥堵概率，1计数器，2这一轮通过数量，3上一轮时间占有率，4这一轮时间占有率]
        p_a_b = p_b_a if p_a_b > 0.1 else 0.1
        self.pool.append([p_a_b, counts, t_counts, p_rate, time_rate])

        print("当前拥堵的概率为：", p_a_b)

        # post
        if p_a_b >= self.threshold and not crowed_block[0]:
            self.pbox = t_boxes
            self.draw()
            push(self.opt, self.im0s, "crowed")
            print("push crowed yes！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            # block
            thread_ptz = Thread(target=start_block, args=(crowed_block,), daemon=True)
            thread_ptz.start()
        self.lock.release()






