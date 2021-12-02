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
from utils.norm import get_area, load_classes, readyaml, func_nms, showfps, getStatus, detect_post_process
from utils.postprocess import postprocess_track
from utils.shape import intersects
from deepsort.utils import _preprocess, _xywh_to_xyxy, _xywh_to_tlwh, _tlwh_to_xyxy, filter_pool, xyxy2xywh,\
    compute_color_for_id, plot_one_box
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from utils.paras import my_yaml
from utils.load_streams import LoadStreams
from camera.utils import create_cameras
from utils.mydraw import show_boxes_draw


SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]

# cameras
cameras = create_cameras()
n_cam = len(cameras)


def detect(opt):
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
    lock = threading.Lock()

    # opt
    rtsps = [cam.rtsp for cam in cameras]
    MODEL_PATH = os.path.join(SRC_PATH, opt.om)
    MODEL_PATH_EX = os.path.join(SRC_PATH, opt.ex)
    print('n_cam', n_cam)
    print('rtsp:', rtsps)
    print("om:", MODEL_PATH)
    # others
    MODEL_WIDTH = opt.width
    MODEL_HEIGHT = opt.height
    NMS_THRESHOLD_CONST = opt.const
    CLASS_SCORE_CONST = opt.myclass
    points = [opt.p0, opt.p1, opt.p2, opt.p3]  # 代表当前点位需要检测的业务

    # Load labels
    names = opt.name
    labels = load_classes(names)
    nc = len(labels)
    print('labels:', labels)
    assert len(labels) > 0, "label file load fail"

    # opencv
    cv2.namedWindow("deepsort", 0)
    cv2.resizeWindow("deepsort", 960, 540)

    # Load model
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)
    model_extractor = Model(MODEL_PATH_EX)

    # Load dataset
    dataset = LoadStreams(rtsps, img_size=(MODEL_WIDTH, MODEL_HEIGHT), n_cam=n_cam)

    # fps
    thread_fps = Thread(target=showfps, args=(vfps,), daemon=True)
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
    areas = [ill_park_areas, people_areas, material_areas]

    limgs = [np.random.random([1, 3, MODEL_WIDTH, MODEL_HEIGHT])] * n_cam

    # 异常停车相似度矩阵
    # 0：先验停车概率
    # 1：长宽比
    # 2：大小
    # 3：7 xyxy
    matrix_park = np.zeros((1920, 1080, 7))
    matrix_park[:, :, 0] = 0.01
    matrixs_park = [matrix_park] * 4
    # histogram
    # histograms = [np.zeros((1920, 1080, 265, 1))] * 4
    histograms = [np.zeros((5, 5))] * 4

    # 开始取流检测--------------------------------------------------------------------------------------------------------
    for i, (img, im0s, nn) in enumerate(dataset):
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

        all_boxes = detect_post_process(infer_output, nc, CLASS_SCORE_CONST)

        if all_boxes.shape[0] > 0:
            # nms
            real_box = func_nms(all_boxes, NMS_THRESHOLD_CONST)
            print("real_box:", real_box.shape)

            # draw
            if opt.show and nn == 3:
                show_boxes_draw(real_box, s_im0s, labels, width, height)
                cv2.imshow('deepsort', s_im0s)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    model.destroy()
                    model_extractor.destroy()
                    print('quit')

            # thread----------------------------------------------------------------------------------------------------
            if len(real_box) > 0:
                # filter
                for pool in pools:
                    filter_pool(pool, POOL_THRES)

                # 第nn个相机的thread
                thread_post = Thread(target=postprocess_track, args=(nn, cameras[nn].ip, points,
                                                                     opt, im0s, real_box,
                                                                     pools, areas, lock,
                                                                     matrixs_park, histograms))
                thread_post.start()

        # fps
        vfps[nn] += 1


if __name__ == '__main__':

    opt = my_yaml()
    detect(opt)




