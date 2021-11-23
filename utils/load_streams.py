import cv2
import time
import numpy as np
from threading import Thread

# load rtsp 多路
class LoadStreams:
    def __init__(self, sources, img_size=608, n_cam=0):
        # init
        self.mode = 'images'
        self.img_size = img_size
        self.imgs = [None]
        self.sources = sources
        self.n_cam = n_cam

        # Start
        caps = []
        for source in sources:
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), 'Failed to open %s' % source
            caps.append(cap)
            # camera info
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            print('success (%gx%g at %.2f FPS).' % (w, h, fps))
        self.caps = caps

        # read
        _, self.imgs = self.caps[0].read()
        self.index = 0

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        # 轮相机取流
        n = self.index
        ret = self.caps[n].grab()

        # 没有帧，则尝试重新连接...
        while not ret:
            print("rtsp重新连接中---------------------------")
            time.sleep(1)
            cap = cv2.VideoCapture(self.sources[n])
            if not cap:
                continue
            self.caps[n] = cap
            ret = cap.grab()

        # fps = 25
        _, self.imgs = self.caps[n].retrieve()

        # judge
        self.index += 1
        if self.index == self.n_cam:
            self.index = 0

        self.count += 1
        img0 = self.imgs.copy()
        print("get a img-------------------------------------------------------------------:", self.index + 1)

        # resize
        img = cv2.resize(img0, self.img_size)
        img = img[np.newaxis, :]
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0

        return img, img0, self.index - 1

    def __len__(self):
        return 0
