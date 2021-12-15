from base_strategy import Strategy


class IiiParkStrategy(Strategy):
    # init
    def __init__(self, url, point, boxes, pool, im0s, labels, height, width, matrix_park,):
        Strategy.__init__(self, url, point, boxes, pool, im0s, labels, height, width)
        self.matrix_park = matrix_park

    def do(self, ):
        pass