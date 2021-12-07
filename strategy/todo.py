import numpy as np

from .strategy import IiiParkStrategy, PeopleStrategy, MaterialStrategy, illegalDriving, crowedSrtategy


def todo(nn, point, c_box, pools, opt, im0s, lock, matrix_park):

    thresholds = [20, 3, 3, 20, 0.8]

    # 不同处理策略集合
    strategies = {
        0: IiiParkStrategy(nn, point, c_box[0], pools[0][nn], opt, im0s, thresholds[0], lock, matrix_park) if c_box[0].size != 0 else 'no',
        # 1: PeopleStrategy(nn, point, c_box[1], pools[1][nn], opt, im0s, thresholds[1], lock) if c_box[1].size != 0 else 'no',
        #2: MaterialStrategy(nn, point, c_box[0], pools[2][nn], opt, im0s, thresholds[2], lock) if c_box[2].size != 0 else 'no',
        #3: illegalDriving(c_box[3], pool[3], opt, im0s, thresholds[3], lock) if c_box[3].size != 0 else 'no',
        #4: crowedSrtategy(c_box[4], pool[4], opt, im0s, thresholds[4], lock) if isinstance(c_box[4], list) else 'no'
    }

    for k, v in strategies.items():
        if v != 'no':
            v.do()