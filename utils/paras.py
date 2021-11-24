import argparse
from utils.norm import readyaml
from utils.myjson import read_json


def my_parser(post, om, ex, name,
              width, height, const, myclass,
              p0, p1, p2, p3):
    parser = argparse.ArgumentParser()
    # area
    parser.add_argument('--post', type=str, default=post)
    parser.add_argument('--om', type=str, default=om)
    parser.add_argument('--ex', type=str, default=ex)
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--show', action='store_true')
    # others
    parser.add_argument('--width', type=str, default=width)
    parser.add_argument('--height', type=str, default=height)
    parser.add_argument('--const', type=str, default=const)
    parser.add_argument('--myclass', type=str, default=myclass)
    # points
    parser.add_argument('--p0', type=str, default=p0)
    parser.add_argument('--p1', type=str, default=p1)
    parser.add_argument('--p2', type=str, default=p2)
    parser.add_argument('--p3', type=str, default=p3)
    opt = parser.parse_args()

    return opt


def my_yaml():
    y = readyaml()
    opt = my_parser(y['POST'], y['OM'], y['EX'], y['NAME'],
                    y['MODEL_WIDTH'], y['MODEL_HEIGHT'], y['NMS_THRESHOLD_CONST'], y['CLASS_SCORE_CONST'],
                    y['p0'], y['p1'], y['p2'], y['p3'])

    return opt
