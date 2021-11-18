import cv2
import time
import numpy as np
from threading import Thread

# load rtsp
class LoadStreams:
    def __init__(self, source='', img_size=608):
        # init
        self.mode = 'images'
        self.img_size = img_size
        self.imgs = [None]
        self.source = source

        # Start
        cap = cv2.VideoCapture(source)
        self.cap = cap
        assert cap.isOpened(), 'Failed to open %s' % source
        # width & height & fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100

        # read
        _, self.imgs = cap.read()

        # thread
        thread = Thread(target=self.update, args=([cap]), daemon=True)
        print('success (%gx%g at %.2f FPS).' % (w, h, fps))
        thread.start()

        print('')  # newline

    def update(self, cap):
        n = 0
        while cap.isOpened():
            n += 1
            ret = cap.grab()

            # 若没有帧返回，则重新刷新rtsp视频流
            while not ret:
                cap = cv2.VideoCapture(self.source)
                if not cap:
                    continue
                ret = cap.grab()
                print("rtsp重新连接中---------------------------")
                time.sleep(0.5)

            # fps = 25
            if n == 2:
                _, self.imgs = cap.retrieve()
                n = 0


    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        print("get a img-----------------------------------------------------:", img0.shape)

        # resize
        img = cv2.resize(img0, self.img_size)
        img = img[np.newaxis, :]
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0

        return self.source, img, img0, self.cap

    def __len__(self):
        return 0
