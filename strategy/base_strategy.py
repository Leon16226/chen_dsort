import abc
from deepsort.utils import compute_color_for_id, plot_one_box


class Strategy(metaclass=abc.ABCMeta):
    def __init__(self, nn, point, boxes, pool, opt, im0s, threshold, lock):
        self.nn = nn
        self.point = point
        self.boxes = boxes
        self.pool = pool
        self.opt = opt
        self.im0s = im0s
        self.pbox = []
        self.threshold = threshold
        self.lock = lock

        # names
        names = opt.name
        with open(names, 'r') as f:
            names = f.read().split('\n')
        self.labels = list(filter(None, names))

    @abc.abstractmethod
    def do(in_area_box):
        pass

    # 画标签
    def draw(self):
        # draw boxes for visualization----------------------------------------------------------------------
        for i, box in enumerate(self.pbox):
            bboxes = box[0:4]
            cls = box[4]
            conf = box[5]
            c = int(cls)

            label = f'{self.labels[c]}{conf:.2f}'
            color = compute_color_for_id(c)
            plot_one_box(bboxes, self.im0s, label=label, color=color, line_thickness=2)