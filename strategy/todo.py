import numpy as np

from .strategy import IiiParkStrategy, PeopleStrategy, MaterialStrategy, illegalDriving, crowedSrtategy


def todo(url, point, boxes, pool, im0s, labels, height, width, matrix_park, matrixs_crowd):

    if len(boxes[0]):
        strategy_of_park = IiiParkStrategy(url, point, boxes[0], pool, im0s, labels, height, width, matrix_park)
        strategy_of_park.do()
    elif len(boxes[4]):
        # strategy_of_people = PeopleStrategy()
        pass


    # # 不同处理策略集合
    # strategies = {
    #     0: IiiParkStrategy(point, c_box[0], pool[0], im0s,
    #                        matrix_park, labels,
    #                        height, width) if c_box[0].size != 0 else 'no',
    #     # 1: PeopleStrategy(nn, point, c_box[1], pools[1][nn], opt, im0s, thresholds[1], lock) if c_box[1].size != 0 else 'no',
    #     #2: MaterialStrategy(nn, point, c_box[0], pools[2][nn], opt, im0s, thresholds[2], lock) if c_box[2].size != 0 else 'no',
    #     #3: illegalDriving(c_box[3], pool[3], opt, im0s, thresholds[3], lock) if c_box[3].size != 0 else 'no',
    #     #4: crowedSrtategy(c_box[4], pool[4], opt, im0s, thresholds[4], lock) if isinstance(c_box[4], list) else 'no'
    # }

