import numpy as np
import yaml
import time
from onvif import ONVIFCamera
import zeep

# 获取检测区域
def get_area(points):
    points = points.split(',')
    p = []
    for i, s in enumerate(points):
        if i % 2 == 0:
            p.append((int(points[i]), int(points[i + 1])))

    return np.array(p)


# 获取类别标签
def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# 读取配置文件
def readyaml():
    f = open('config.yaml', 'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    return x


# 检测框后处理
def detect_post_process(infer_output, nc, CLASS_SCORE_CONST):
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

    return all_boxes

# nms
def func_nms(boxes, nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2] - boxes[:, 0]
    b_h = boxes[:, 3] - boxes[:, 1]

    areas = b_w * b_h

    scores = boxes[:, 5]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        i_other = order[1:]

        # inter area  : left_top   right_bottom
        xx1 = np.maximum(b_x[i], b_x[i_other])
        yy1 = np.maximum(b_y[i], b_y[i_other])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[i_other] + b_w[i_other])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[i_other] + b_h[i_other])
        # inter area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        # calc IoU
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        IoU = inter / union

        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]

    final_boxes = np.array([boxes[i] for i in keep])
    return final_boxes

# fps
def showfps(vfps):
    print("rtsp success")
    while(True):
        time.sleep(1.0)
        for i, fps in enumerate(vfps):
            print("fps:", i, ":", vfps[i])
            vfps[i] = 0



# ptz
def getStatus(ptz_gate):
    print("ptz success")

    def zeep_pythonvalue(self, xmlvalue):
        return xmlvalue

    while True:
        # init
        zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
        mycam = ONVIFCamera("10.33.27.45", 80, "admin", "admin12345")
        media = mycam.create_media_service()
        # info
        resp = mycam.devicemgmt.GetHostname()
        print('My camera hostname:' + str(resp.Name))
        # ptz service
        ptz = mycam.create_ptz_service()
        params = ptz.create_type('GetStatus')
        # profile
        media_profile = media.GetProfiles()[0]
        params.ProfileToken = media_profile.token
        res = ptz.GetStatus(params)
        print("ptz")
        p = res['Position']['PanTilt']['x']
        t = res['Position']['PanTilt']['y']
        z = res['Position']['Zoom']['x']

        if p == 0.566611 and t == -0.220222 and z == 0.0:
            ptz_gate[0] = True
        else:
            ptz_gate[0] = False

        # 比较

        time.sleep(1)


def start_block(crowed_block):
    # 5分钟
    crowed_block[0] = True
    time.sleep(300)
    crowed_block[0] = False

