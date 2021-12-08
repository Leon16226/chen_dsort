import argparse
import os
import shutil
import threading
import time
import cv2
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

SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
LOCAL_IP = '192.168.1.149'


STATES = {'stop': False, 'start': False, 'restart': False}
PATH = {'eventUploadPath': '',
        'trafficUploadPath': '',
        'configsPath': '',
        'carNoUploadPath': ''}
DETECT_LIVE = [False]
IP_STATES = {}


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
        self.points = [self.opt.p0, self.opt.p1, self.opt.p2, self.opt.p3]
        self.lock = threading.Lock()

        # Load labels
        names = self.opt.name
        self.labels = load_classes(names)
        self.nc = len(self.labels)
        print('labels:', self.labels)
        assert len(self.labels) > 0, "label file load fail"

        # opencv
        cv2.namedWindow("deepsort", 0)
        cv2.resizeWindow("deepsort", 960, 540)

        self.model = None
        self.dataset = None
        self.myfps = None

        print('detection 初始化完毕...')

    def detect(self,):
        # init model
        model_path = os.path.join(SRC_PATH, self.opt.om)
        print("om1:", model_path)
        # Load model
        model = Model(model_path)

        # cameras
        cameras = create_cameras_online(self.path.get('configsPath'), LOCAL_IP)
        rtsps = [cam.rtsp for cam in cameras]
        n_cam = len(cameras)
        print('n_cam', n_cam)
        print('rtsp:', rtsps)

        for cam in cameras:
            IP_STATES[cam.get_ip()] = True

        # 属于方法的局部变量
        vfps = [0] * n_cam
        ptz_gate = [True] * n_cam
        crowed_block = [False] * n_cam

        # pool
        POOL_THRES = 20
        car_id_pool = [[]] * n_cam
        people_id_pool = [[]] * n_cam
        material_id_pool = [[]] * n_cam
        illdri_id_pool = [[]] * n_cam
        crowed_id_pool = [[]] * n_cam
        pools = [car_id_pool, people_id_pool, material_id_pool,
                 illdri_id_pool, crowed_id_pool]

        # Load dataset
        self.dataset = LoadStreams(rtsps, img_size=(self.model_width, self.model_height), n_cam=n_cam)

        # fps
        self.myfps = Fps()
        thread_fps = Thread(target=self.myfps.showfps, args=(vfps,), daemon=True)
        thread_fps.start()

        # ptz
        # thread_ptz = Thread(target=getStatus, args=(ptz_gate[0],), daemon=True)
        # thread_ptz.start()

        # 包括n_cam个相机的检测区域
        # 1. 异常停车区域
        # 2. 行人检测区域
        # 3. 抛撒物区域
        ill_park_areas = [cam.get_ill_park() for cam in cameras]
        people_areas = [cam.get_people() for cam in cameras]
        material_areas = [cam.get_material() for cam in cameras]
        areas = {'IllegalPark': ill_park_areas,
                 'People': people_areas,
                 'ThrowThings': material_areas}

        limgs = [np.random.random([1, 3, self.model_width, self.model_height])] * n_cam

        # 异常停车相似度矩阵
        # 0：先验停车概率
        # 1：长宽比
        # 2：大小
        # 3,4,5,6 xyxy
        # 7 锁
        matrix_park = np.zeros((1920, 1080, 8))
        matrix_park[:, :, 0] = 0.01
        matrixs_park = [matrix_park] * 4
        # histogram
        # histograms = [np.zeros((1920, 1080, 265, 1))] * 4
        # histograms = [np.zeros((5, 5))] * 4

        # 开始取流检测--------------------------------------------------------------------------------------------------------
        for i, (img, im0s, nn) in enumerate(self.dataset):
            # 控制
            if self.states['restart']:
                print('停止检测...')
                for cam in cameras:
                    IP_STATES[cam.get_ip()] = False
                break

            s_im0s = im0s.copy()

            # 情况1：重复帧
            if np.sum(limgs[nn] - img) == 0:
                print("xxxxxxxxxxxxxxxxx跳过这帧xxxxxxxxxxxxxxxxx")
                continue
            limgs[nn] = img

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
                if self.opt.show and nn == 1:
                    show_boxes_draw(real_box, s_im0s, self.labels, width, height)
                    cv2.imshow('deepsort', s_im0s)
                    if cv2.waitKey(1) == ord('q'):
                        print('停止检测...')
                        break

                # thread----------------------------------------------------------------------------------------------------
                if len(real_box) > 0 and nn == 1:
                    # filter
                    for pool in pools:
                        filter_pool(pool, POOL_THRES)

                    # 第nn个相机的thread
                    thread_post = Thread(target=postprocess_track, args=(nn, cameras[nn].ip, self.points,
                                                                         self.opt, im0s, real_box,
                                                                         pools, areas, self.lock,
                                                                         matrixs_park,))
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


if __name__ == '__main__':
    start_socket()
