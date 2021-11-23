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
from utils.norm import get_area, load_classes, readyaml, func_nms, showfps, getStatus
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
    id_thres = 20
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
    ill_park_areas = [cam.get_ill_park() for cam in cameras]
    areas = [ill_park_areas]

    limgs = [np.random.random([1, 3, MODEL_WIDTH, MODEL_HEIGHT])] * n_cam
    # 开始取流检测--------------------------------------------------------------------------------------------------------
    for i, (img, im0s, nn) in enumerate(dataset):

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

        # 模型输出box的数量
        MODEL_OUTPUT_BOXNUM = infer_output.shape[1]

        # 转换处理并根据置信度门限过滤box
        result_box = infer_output[:, :, 0:6].reshape((-1, 6))
        list_class = infer_output[:, :, 5:5 + nc].reshape((-1, nc))
        # class
        list_max = list_class.argmax(axis=1).reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 4] = list_max[:, 0]
        # conf
        list_max = list_class.max(axis=1).reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]
        all_boxes = result_box[result_box[:, 5] >= CLASS_SCORE_CONST]

        if all_boxes.shape[0] > 0:
            # 1.根据nms过滤box
            real_box = func_nms(all_boxes, NMS_THRESHOLD_CONST)
            print("real_box:", real_box.shape)

            # draw
            if opt.show:
                for box in real_box:
                    bboxes = box[0:4]
                    cls = box[4]
                    conf = box[5]
                    cls = int(cls)

                    label = f'{labels[cls]}{conf:.2f}'
                    color = compute_color_for_id(cls)
                    plot_one_box(bboxes, im0s, label=label, color=color, line_thickness=2)

            # thread----------------------------------------------------------------------------------------------------
            if len(real_box) > 0:
                # filter
                for pool in pools:
                    filter_pool(pool, id_thres)

                # 当前nn个相机的thread
                thread_post = Thread(target=postprocess_track, args=(nn,
                                                                     opt, im0s, real_box,
                                                                     pools, areas, lock))
                thread_post.start()

        # fps
        vfps[nn] += 1

        # show----------------------------------------------------------------------------------------------------------
        if opt.show and nn == 0:
            cv2.imshow("deepsort", im0s)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                model.destroy()
                model_extractor.destroy()
                print('quit')
                break


if __name__ == '__main__':

    opt = my_yaml()
    detect(opt)




