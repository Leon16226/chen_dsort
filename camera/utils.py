from .my_camera import Camera
from net.get import get_point_json
from utils.myjson import read_json
import os


def create_cameras():
    cameras = []
    path = 'f_json/'
    files = os.listdir(path)
    for i, file in enumerate(files):
        file = path + file
        myjson = read_json(file)
        cam = Camera(myjson)
        cameras.append(cam)
        print('第', i+1, '个json配置文件解析完成')

    return cameras


def create_cameras_online(ip, port, server):
    cameras = []
    myjson = get_point_json(ip, port, server)
    for i, json in enumerate(myjson):
        cam = Camera(json)
        cameras.append(cam)
        print('第', i + 1, '个json配置文件解析完成')

    return cameras




if __name__ == '__main__':
    cameras = create_cameras()

