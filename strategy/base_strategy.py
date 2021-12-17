import abc
import threading

from deepsort.utils import compute_color_for_id, plot_one_box


class Strategy(metaclass=abc.ABCMeta):
    def __init__(self, url, point, boxes, pool, im0s, labels, height, width):
        self.url = url
        self.point = point
        self.boxes = boxes
        self.pool = pool
        self.im0s = im0s
        self.pbox = []
        self.labels = labels
        self.height = height
        self.width = width
        self.lock = threading.Lock()

    @abc.abstractmethod
    def do(self):
        pass

    # 画标签
    def draw(self):
        # draw boxes for visualization----------------------------------------------------------------------
        my_im0s = self.im0s.copy()
        for i, box in enumerate(self.pbox):
            bboxes = box[0:4]
            bboxes[[0, 2]] = bboxes[[0, 2]] * self.width
            bboxes[[1, 3]] = bboxes[[1, 3]] * self.height
            cls = box[4]
            conf = box[5]
            c = int(cls)

            label = f'{self.labels[c]}{conf:.2f}'
            color = compute_color_for_id(c)
            plot_one_box(bboxes, my_im0s, label=label, color=color, line_thickness=2)