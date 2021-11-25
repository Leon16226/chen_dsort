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
from utils.paras import my_yaml, my_json
from utils.load_streams import LoadStreams


SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]

vfps = [0]
ptz_gate = [True]

# pool
id_thres = 20
car_id_pool = []
people_id_pool = []
material_id_pool = []
illdri_id_pool = []
crowed_id_pool = []
lock = threading.Lock()

# block
crowed_block = [False]

rtsps = ('rtsp://192.168.1.20/yctc7.mp4',
         'rtsp://192.168.1.20/xr1.mp4',
         'rtsp://192.168.1.20/psw3.mp4',
         'rtsp://192.168.1.20/ycxs1.mp4',
         'rtsp://192.168.1.20/hx3.mp4')

points = ('832,151,901,145,1656,955,169,925',
          '1626,313,1753,327,1716,1002,1009,823',
          '1048,100,1255,100,1202,640,259,570',
          '1108,300,1161,301,1218,1063,961,1068',
          '623,47,706,47,858,198,488,198')

side = ('1125,211,1153,211,1195,995,963,995',)
crowed = ('643,52,663,52,673,155,601,155',)
tcrowed = ('637,70,664,70,667,100,624,100',)

false_side = ('0,0,10,0,10,10,0,10',)
false_crowed = ('0,0,10,0,10,10,0,10',)
false_tcrowed = ('0,0,10,0,10,10,0,10',)

def detect(opt):
    # show
    num = opt.num

    # opt
    rtsp = rtsps[num]
    MODEL_PATH = os.path.join(SRC_PATH, opt.om)
    MODEL_PATH_EX = os.path.join(SRC_PATH, opt.ex)
    print('rtsp:', rtsp)
    print("om:", MODEL_PATH)
    # others
    MODEL_WIDTH = opt.width
    MODEL_HEIGHT = opt.height
    NMS_THRESHOLD_CONST = opt.const
    CLASS_SCORE_CONST = opt.myclass
    MODEL_OUTPUT_BOXNUM = 10647
    # deepsort
    MAX_DIST = opt.dist
    MIN_CONFIDENCE = opt.dconst
    NMS_MAX_OVERLAP = opt.overlap
    MAX_IOU_DISTANCE = opt.distance
    MAX_AGE = opt.age
    N_INIT = opt.ninit
    NN_BUDGET = opt.nbudget

    # global
    global vfps
    global ptz_gate
    global car_id_pool
    global people_id_pool
    global material_id_pool
    global crowed_block

    # 1. 检测区域
    # 2. 异常行驶区域
    # 3. 缓行区域

    # 0: 异常停车
    # 1：行人
    # 2：抛撒物
    # 3：异常行驶 有side
    # 4：缓行
    if num == 0:
        point1 = get_area(points[num])
        point2 = get_area(false_side[0])
        point3 = get_area(false_crowed[0])
        p_crowed_time = get_area(false_tcrowed[0])
    elif num == 1:
        point1 = get_area(points[num])
        point2 = get_area(false_side[0])
        point3 = get_area(false_crowed[0])
        p_crowed_time = get_area(false_tcrowed[0])
    elif num == 2:
        point1 = get_area(points[num])
        point2 = get_area(false_side[0])
        point3 = get_area(false_crowed[0])
        p_crowed_time = get_area(false_tcrowed[0])
    elif num == 3:
        point1 = get_area(points[num])
        point2 = get_area(side[0])
        point3 = get_area(false_crowed[0])
        p_crowed_time = get_area(false_tcrowed[0])
    elif num == 4:
        point1 = get_area(points[num])
        point2 = get_area(false_crowed[0])
        point3 = get_area(crowed[0])
        p_crowed_time = get_area(tcrowed[0])

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
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # deepsort init
    max_cosine_distance = MAX_DIST
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=max_cosine_distance, max_age=MAX_AGE, n_init=N_INIT)

    # fps
    thread_fps = Thread(target=showfps, args=(vfps,), daemon=True)
    thread_fps.start()

    # ptz
    thread_ptz = Thread(target=getStatus, args=(ptz_gate[0],), daemon=True)
    thread_ptz.start()

    limg = np.random.random([1, 3, MODEL_WIDTH, MODEL_HEIGHT])
    # 开始取流检测--------------------------------------------------------------------------------------------------------
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):

        s_im0s = im0s.copy()

        # 情况1：重复帧
        if np.sum(limg - img) == 0:
            print("xxxxxxxxxxxxxxxxx跳过这帧xxxxxxxxxxxxxxxxx")
            continue
        limg = img

        # 情况2：ptz
        if not ptz_gate:
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

            # 2.保留在检测区域内的框或者拥堵区域
            dets = []
            crows = []
            car_num = 0
            for i, box in enumerate(real_box):
                center = np.array([(box[0] + box[2]) / 2 * width, (box[1] + box[3]) / 2 * height])
                if intersects(center, point1):
                    dets.append(box)
                    if intersects(center, point3):
                        crows.append(box)
                    # 确定检测区域一共有几辆车
                    car_ids = [0, 1, 13, 14, 16, 17]
                    if box[4] in car_ids:
                        car_num += 1
            det = np.array(dets)
            crows = np.array(crows)
            if len(crows) != 0 and not crowed_block[0]:
                crows[:, [0, 2]] = (crows[:, [0, 2]] * width).round()
                crows[:, [1, 3]] = (crows[:, [1, 3]] * height).round()
            print("det_box:", len(det))
            print("crows_box", len(crows))
            print("car_num", car_num)

            # 开始跟踪的处理-----------------------------------------------------------------------------------------------
            if det is not None and len(det):
                det[:, [0, 2]] = (det[:, [0, 2]] * width).round()
                det[:, [1, 3]] = (det[:, [1, 3]] * height).round()
                print("det f:", det[:, :4].shape)

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 5]
                clss = det[:, 4]
                print("xywhs:", xywhs.shape)

                # crops
                im_crops = []
                for i, box in enumerate(xywhs):
                    x1, y1, x2, y2 = _xywh_to_xyxy(box, height, width)
                    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
                        break
                    im = im0s[y1:y2, x1:x2]
                    im_crops.append(im)

                # features
                if im_crops:
                    im_batch = _preprocess(im_crops)
                    features = model_extractor.execute([im_batch, np.array(im_batch.shape)], 'deepsort')
                    print("features:", features[0].shape)
                    features = features[0][0:im_batch.shape[0], :]
                else:
                    features = np.array([])


                if features.shape[0] > 0:
                    # Detections
                    bbox_tlwh = _xywh_to_tlwh(xywhs)
                    detections = [Detection(bbox_tlwh[i], confs[i], feature) for i, feature in enumerate(features)]

                    # update tracker
                    tracker.predict()
                    tracker.update(detections, clss)

                    # output
                    outputs = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        box = track.to_tlwh()
                        x1, y1, x2, y2 = _tlwh_to_xyxy(box, height, width)
                        track_id = track.track_id
                        class_id = track.class_id
                        conf = track.conf
                        outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf], dtype=np.int))

                    # draw
                    if opt.show:
                        for i, box in enumerate(outputs):
                            bboxes = box[0:4]
                            id = box[4]
                            cls = box[5]
                            c = int(cls)
                            # conf = box[6]

                            label = f'{id} {labels[c]}'
                            color = compute_color_for_id(id)
                            plot_one_box(bboxes, s_im0s, label=label, color=color, line_thickness=2)

                    # thread--------------------------------------------------------------------------------------------
                    if len(outputs) > 0:
                        # 保持pool为一定大小否则内存溢出
                        car_id_pool = filter_pool(car_id_pool, id_thres)
                        people_id_pool = filter_pool(people_id_pool, id_thres)
                        material_id_pool = filter_pool(material_id_pool, id_thres)

                        # thread
                        thread_post = Thread(target=postprocess_track, args=(outputs,
                                                                             car_id_pool, people_id_pool,
                                                                             material_id_pool, illdri_id_pool,
                                                                             crowed_id_pool,
                                                                             opt, im0s,
                                                                             lock,
                                                                             point2,
                                                                             point3,
                                                                             crows,
                                                                             p_crowed_time,
                                                                             crowed_block,
                                                                             car_num))
                        thread_post.start()
            else:
                tracker.increment_ages()

        # fps-----------------------------------------------------------------------------------------------------------
        vfps[0] += 1

        # show----------------------------------------------------------------------------------------------------------
        if opt.show:
            point_s = point1.reshape((-1, 1, 2))
            cv2.polylines(s_im0s, [point_s], True, (0, 255, 255))

            cv2.imshow("deepsort", s_im0s)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                model.destroy()
                model_extractor.destroy()
                print('quit')
                break


if __name__ == '__main__':
    json = False

    if json:
        opt = my_json()
    else:
        opt = my_yaml()

    detect(opt)




