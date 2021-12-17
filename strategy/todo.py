import numpy as np

from .strategy import IiiParkStrategy
from .strategy_crowded import CrowedStrategy


def todo(url, point, boxes, pool, im0s, labels, height, width, matrix_park, matrix_crowd, area_of_crowd):

    if len(boxes[0]):
        strategy_of_park = IiiParkStrategy(url, point, boxes[0], pool, im0s, labels, height, width, matrix_park)
        strategy_of_park.do()
    elif len(boxes[4]):
        strategy_of_crowed = CrowedStrategy(url, point, boxes[4], pool, im0s, labels, height, width, matrix_crowd, area_of_crowd)
        strategy_of_crowed.do()

