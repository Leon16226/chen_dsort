from .my_camera import Camera
import os


def create_cameras():
    cameras = []
    path = 'f_json/'
    files = os.listdir(path)
    for i, file in enumerate(files):
        file = path + file
        cam = Camera(file)
        cameras.append(cam)
        print('第', i+1, '个json配置文件解析完成')

    return cameras


if __name__ == '__main__':
    cameras = create_cameras()

