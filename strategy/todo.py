import numpy as np

from .strategy import CarStrategy, PeopleStrategy, MaterialStrategy, illegalDriving, crowedSrtategy


def todo(c_box, pool, opt, im0s, lock):

    thresholds = [20, 3, 3, 5, 0.70]

    # 不同处理策略集合
    strategies = {
        0: CarStrategy(c_box[0], pool[0], opt, im0s, thresholds[0], lock) if c_box[0].size != 0 else 'no',
        1: PeopleStrategy(c_box[1], pool[1], opt, im0s, thresholds[1], lock) if c_box[1].size != 0 else 'no',
        2: MaterialStrategy(c_box[2], pool[2], opt, im0s, thresholds[2], lock) if c_box[2].size != 0 else 'no',
        3: illegalDriving(c_box[3], pool[3], opt, im0s, thresholds[3], lock) if c_box[3].size != 0 else 'no',
        4: crowedSrtategy(c_box[4], pool[4], opt, im0s, thresholds[4], lock) if isinstance(c_box[4], list) else 'no'
    }

    for k, v in strategies.items():
        if v != 'no':
            v.do()