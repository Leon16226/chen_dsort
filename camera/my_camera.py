from utils.myjson import read_json
from .area import Area


class Camera:
    def __init__(self, mjson):
        # json


        obj = mjson

        # 相机基本信息
        self.ip = obj['ip']
        self.manufacturer = obj['manufacturer']  # 厂家
        self.name = obj['name']  # 桩号
        self.password = obj['password']  # 密码
        self.rtsp = obj['rtsp']
        self.scene = obj['scene']   # 场景
        self.subScene = obj['subScene']  # 子场景
        self.type = obj['type']  # 类型
        self.userName = obj['userName']

        # 点位配置
        self.config = obj['configs']['config'][0]
        self.prestaging = self.config['prestaging']  # 预置位0或1
        self.ptz = (0, 0, 0)
        self.road = self.config['road'][0]
        # 检测区域
        self.detectAreas = []
        for i, info in enumerate(self.road['detectAreas']):
            area = Area(info)
            self.detectAreas.append(area)

    def get_ill_park(self):
        areas = []
        for area in self.detectAreas:
            if 'IllegalPark' in area.event_check:
                areas.append(area.points)
        return areas

    def get_people(self):
        areas = []
        for area in self.detectAreas:
            if 'People' in area.event_check:
                areas.append(area.points)
        return areas

    def get_material(self):
        areas = []
        for area in self.detectAreas:
            if 'ThrowThings' in area.event_check:
                areas.append(area.points)
        return areas













