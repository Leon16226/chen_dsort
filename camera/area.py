
class Area:
    def __init__(self, area):
        self.area = area
        self.points = self.area['points']
        self.event_check = []  # 此区域检测事件
        for k in self.area.keys:
            if k != 'points' and k != 'paintType':
                self.event_check.append(k)

