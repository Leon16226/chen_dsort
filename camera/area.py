

# 只有两块重要信息
# 1. 一个检测区域
# 2. 检测业务
class Area:
    def __init__(self, area):
        self.area = area
        self.points = self.area['points']
        self.event_check = []  # 此区域检测业务
        for k in self.area.keys():
            if k != 'points' and k != 'paintType':
                self.event_check.append(k)

