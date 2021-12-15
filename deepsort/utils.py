import cv2
import numpy as np

# id pool
def filter_pool(pool, threshold):
    return pool[-threshold:] if len(pool) > threshold else pool


# iou
def iou(box1, box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    xa, ya, xb, yb = box2[0], box2[1], box2[2], box2[3]

    inter = (np.minimum(x2, xb) - np.maximum(x1, xa)) * (np.minimum(y2, yb) - np.maximum(y1, ya))

    # Union Area
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = xb - xa, yb - ya
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou

    return iou


# iou ndarray
def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return float(intersect / union)

# angle
def calc_angle(car_c, people_c):
    if not isinstance(car_c, np.ndarray):
        car_c = np.array(car_c)
    if not isinstance(people_c, np.ndarray):
        people_c = np.array(people_c)

    relative_p = people_c - car_c.reshape((-1, 1))



# rescale boxes
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2   # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2   # y center
    y[:, 2] = (x[:, 2] - x[:, 0])   # width
    y[:, 3] = (x[:, 3] - x[:, 1])   # height
    return y


def _xyxy_to_tlwh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h


def _xywh_to_xyxy(bbox_xywh, height, width):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2


# xc_yc_w_h to xtl_ytl_w_h
def _xywh_to_tlwh(bbox_xywh):
        bbox_tlwh = bbox_xywh.copy()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh


def _tlwh_to_xyxy(bbox_tlwh, height, width):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), height - 1)
        return x1, y1, x2, y2


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# preprocess
size = (64, 128)
# norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])

def norm(im):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # output = im.copy()
    for i in np.arange(0, 3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im


def _preprocess(im_crops):
    """
    TODO:
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
    """

    def _resize(im, size):
        return cv2.resize(im.astype(np.float32) / 255., size)

    im_batch = np.concatenate([np.expand_dims(norm(_resize(im, size)).transpose(2, 0, 1), axis=0) for im in im_crops], axis=0)
    # im_batch = [np.expand_dims(norm(_resize(im, size)), axis=0) for im in im_crops]
    return im_batch