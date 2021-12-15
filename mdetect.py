import argparse
import os
import shutil
import threading
import time
import cv2
import numpy
import numpy as np
from threading import Thread
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
from utils.norm import get_area, load_classes, readyaml, func_nms, Fps, getStatus, detect_post_process
from utils.postprocess import postprocess_track
from utils.shape import intersects
from deepsort.utils import _preprocess, _xywh_to_xyxy, _xywh_to_tlwh, _tlwh_to_xyxy, filter_pool, xyxy2xywh, \
    compute_color_for_id, plot_one_box
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from utils.paras import my_yaml
from utils.load_streams import LoadStreams
from camera.utils import create_cameras, create_cameras_online
from utils.mydraw import show_boxes_draw
from mysocket.my_socket import My_Socket
from strategy.img_similarity import sim_dhash, calculate_hanming

SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
LOCAL_IP = '192.168.1.149'


STATES = {'stop': False, 'start': False, 'restart': False}  # 命令状态
PATH = {'eventUploadPath': '',
        'trafficUploadPath': '',
        'configsPath': '',
        'carNoUploadPath': ''}
DETECT_LIVE = [False]  # 检测程序状态
IP_STATES = {}  # 点位检测状态

my_show = True
my_ptz = False
my_fps = False


class MyDetection(object):

    def __init__(self,):
        # socket
        self.states = STATES
        self.path = PATH

        self.detect_live = DETECT_LIVE
        self.opt = my_yaml()

        # base config
        self.model_width = self.opt.width
        self.model_height = self.opt.height
        self.nms_threshold_const = self.opt.const
        self.class_score_const = self.opt.myclass

        # Load labels
        names = self.opt.name
        self.labels = load_classes(names)
        self.nc = len(self.labels)
        print('labels:', self.labels)
        assert len(self.labels) > 0, "label file load fail"

        # opencv
        if my_show:
            cv2.namedWindow("deepsort", 0)
            cv2.resizeWindow("deepsort", 960, 540)

        self.model = None
        self.dataset = None
        self.myfps = None

        print('detection 初始化完毕...')

    def detect(self,):
        # load model
        model_path = os.path.join(SRC_PATH, self.opt.om)
        print("om1:", model_path)
        model = Model(model_path)

        # cameras
        cameras = create_cameras_online(self.path.get('configsPath'), LOCAL_IP)
        rtsps = [cam.rtsp for cam in cameras]
        n_cam = len(cameras)
        print('n_cam', n_cam)
        print('rtsp:', rtsps)

        for cam in cameras:
            IP_STATES[cam.get_ip()] = True

        # 点位检测业务
        matrix_of_service = np.zeros((n_cam, 5), dtype=numpy.int8)
        matrix_of_service[1, 0] = 1
        matrix_of_service[2, 4] = 1

        # 属于方法的局部变量
        vfps = [0] * n_cam
        ptz_gate = [True] * n_cam
        crowed_block = [False] * n_cam

        # pool
        dict_strategy = {}

        # Load dataset
        self.dataset = LoadStreams(rtsps, img_size=(self.model_width, self.model_height), n_cam=n_cam)

        # fps
        if my_fps:
            self.myfps = Fps()
            thread_fps = Thread(target=self.myfps.showfps, args=(vfps,), daemon=True)
            thread_fps.start()

        # ptz
        if my_ptz:
            thread_ptz = Thread(target=getStatus, args=(ptz_gate[0],), daemon=True)
            thread_ptz.start()

        # 每个点位的检测区域是固定的
        # 1. 异常停车区域
        # 2. 行人检测区域
        # 3. 抛撒物区域
        points_detect_areas = []
        for cam in cameras:
            point = {}

            area_of_park = cam.get_ill_park()
            area_of_people = cam.get_people()
            area_of_material = cam.get_material()

            point['IllegalPark'] = area_of_park
            point['People'] = area_of_people
            point['ThrowThings'] = area_of_material

            points_detect_areas.append(point)

        hashs = [np.zeros(64, dtype=np.uint8) for n in range(n_cam)]

        # 异常停车状态矩阵 5.64MB
        # 0：先验停车概率
        # 1：长宽比
        # 2：大小
        # 3,4,5,6 xyxy
        # 7 锁
        matrixs_park = [np.zeros((608, 608, 8), dtype=numpy.float16) if matrix_of_service[n, 0] == 1 else [] for n in range(n_cam)]
        # 缓行状态矩阵
        # 0 ：过去30秒此像素点的空间占有率
        # 1 ：上一轮的空间占有率
        # 2 ：这一轮的第几帧
        matrixs_crowd = [np.zeros((608, 608, 2), dtype=numpy.float16) if matrix_of_service[n, 4] == 1 else [] for n in range(n_cam)]

        # 开始取流检测----------------------------------------------------------------------------------------------------
        for i, (img, im0s, nn) in enumerate(self.dataset):
            # img: 4.23 MB
            # im0s: 24.47 MB
            # 控制
            if self.states['restart']:
                print('停止检测...')
                for cam in cameras:
                    IP_STATES[cam.get_ip()] = False
                break

            # 情况1：重复帧
            img_hash = sim_dhash(im0s)
            if calculate_hanming(hashs[nn], img_hash) == 0:
                print("xxxxxxxxxxxxxxxxx跳过这帧xxxxxxxxxxxxxxxxx")
                continue
            hashs[nn] = img_hash

            # 情况2：ptz
            if not ptz_gate[nn]:
                print("不在预置位")
                print("xxxxxxxxxxxxxxxxx跳过这帧xxxxxxxxxxxxxxxxx")
                IP_STATES[cameras[nn].get_ip()] = False
                continue

            # 长宽
            height, width = im0s.shape[0], im0s.shape[1]

            # 模型推理-------------------------------------------------------------------------------------------------------
            infer_output = model.execute([img])
            assert infer_output[0].shape[1] > 0, "model no output, please check"
            infer_output_1 = infer_output[1].reshape((1, -1, 4))
            infer_output_2 = np.ones([1, infer_output_1.shape[1], 1])
            infer_output = np.concatenate((infer_output_1,
                                           infer_output_2,
                                           infer_output[0]), axis=2)

            all_boxes = detect_post_process(infer_output, self.nc, self.class_score_const)

            if all_boxes.shape[0] > 0:
                # nms
                real_box = func_nms(all_boxes, self.nms_threshold_const)
                print("real_box:", real_box.shape)

                # draw
                s_im0s = None
                if my_show and nn == 1:
                    s_im0s = im0s.copy()
                    show_boxes_draw(real_box.copy(), s_im0s, self.labels, width, height)
                    cv2.imshow('deepsort', s_im0s)
                    if cv2.waitKey(1) == ord('q'):
                        print('停止检测...')
                        break

                # thread----------------------------------------------------------------------------------------------------
                if len(real_box) > 0 and nn == 1:

                    # 第nn个相机的thread
                    load_url = PATH['eventUploadPath']
                    ip = cameras[nn].ip
                    detect_areas = points_detect_areas[nn]
                    matrix_park = matrixs_park[nn]
                    matrix_service = matrix_of_service[nn]
                    thread_post = Thread(target=postprocess_track, args=(load_url, ip, dict_strategy,
                                                                         height, width, self.labels,
                                                                         im0s, real_box,
                                                                         detect_areas,
                                                                         matrix_park, matrixs_crowd,
                                                                         matrix_service))
                    thread_post.start()

            # fps
            vfps[nn] += 1

        self.set_live(False)
        model.destroy()

    def __del__(self):
            cv2.destroyAllWindows()
            if self.myfps is not None:
                self.myfps.terminate()
            if self.dataset is not None:
                self.dataset.terminate()
            print('quit')

    def get_states(self,):
        return self.states

    def get_live(self):
        return self.detect_live[0]

    def set_live(self, live):
        self.detect_live[0] = live


# start-----------------------------------------------------------------------------------------------------------------
def start_socket():
    # 创建日志文件
    file = open('roadClose.log', 'w')
    file.close()

    print('创建detection对象...')
    # 开始socket服务...
    my_socket = My_Socket()
    thread_socket = Thread(target=my_socket.create_socket_service,
                           args=(LOCAL_IP, 4000, STATES, PATH, DETECT_LIVE, IP_STATES))
    thread_socket.start()
    detection = MyDetection()

    # acl初始化
    acl_resource = AclResource()
    acl_resource.init()

    while True:
        detection = MyDetection() if detection is None else detection
        time.sleep(1)
        print('等待命令...')
        if detection.get_states()['restart']:
            print('开始检测....')
            detection.set_live(True)
            detection.get_states()['restart'] = False
            detection.detect()
            detection = None


