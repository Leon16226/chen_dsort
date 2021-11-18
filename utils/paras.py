import argparse
from utils.norm import readyaml
from utils.myjson import read_json


def my_parser(area, side, crowed, tcrowed,rtsp, post, point, om, ex, name,
              width, height, const, myclass,
              dist, dconst, overlap, distance, age, ninit, nbudget):
    parser = argparse.ArgumentParser()
    # area
    parser.add_argument('--area', type=str, default=area)
    parser.add_argument('--side', type=str, default=side)
    parser.add_argument('--crowed', type=str, default=crowed)
    parser.add_argument('--tcrowed', type=str, default=tcrowed)
    parser.add_argument('--rtsp', type=str, default=rtsp)
    parser.add_argument('--post', type=str, default=post)
    parser.add_argument('--point', type=str, default=point)
    parser.add_argument('--om', type=str, default=om)
    parser.add_argument('--ex', type=str, default=ex)
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--show', action='store_true')
    # others
    parser.add_argument('--width', type=str, default=width)
    parser.add_argument('--height', type=str, default=height)
    parser.add_argument('--const', type=str, default=const)
    parser.add_argument('--myclass', type=str, default=myclass)
    # deepsort
    parser.add_argument('--dist', type=str, default=dist)
    parser.add_argument('--dconst', type=str, default=dconst)
    parser.add_argument('--overlap', type=str, default=overlap)
    parser.add_argument('--distance', type=str, default=distance)
    parser.add_argument('--age', type=str, default=age)
    parser.add_argument('--ninit', type=str, default=ninit)
    parser.add_argument('--nbudget', type=str, default=nbudget)

    opt = parser.parse_args()

    return opt


def my_yaml():
    y = readyaml()
    opt = my_parser(y['AREA'], y['AREA_SIDE'], y['AREA_CROWED'], y['AREA_CROWED_TIME'],
                    y['RTSP'], y['POST'], y['POINT'], y['OM'], y['EX'], y['NAME'],
                    y['MODEL_WIDTH'], y['MODEL_HEIGHT'], y['NMS_THRESHOLD_CONST'], y['CLASS_SCORE_CONST'],
                    y['MAX_DIST'], y['MIN_CONFIDENCE'], y['NMS_MAX_OVERLAP'], y['MAX_IOU_DISTANCE'],
                    y['MAX_AGE'], y['N_INIT'], y['NN_BUDGET'])

    return opt


def my_json():
    json_obj = read_json('/f_json/33.64.37.130.json')
    y = readyaml()

    # detect areas
    detecAreas = json_obj['configs']['config']['road']['detectAreas']

    ThrowThings = detecAreas[7]
    People = detecAreas[8]
    IllegalPark = detecAreas[10]
    IllegalDriving = detecAreas[11]
    Crowed = [detecAreas[18], detecAreas[19],
              detecAreas[20], detecAreas[21],
              detecAreas[22], detecAreas[23]]


    opt = my_parser(y['AREA'], y['AREA_SIDE'], y['AREA_CROWED'],
                    json_obj['rtsp'], y['JSON_POST'], json_obj['ip'], y['OM'], y['EX'], y['NAME'],
                    y['MODEL_WIDTH'], y['MODEL_HEIGHT'], y['NMS_THRESHOLD_CONST'], y['CLASS_SCORE_CONST'],
                    y['MAX_DIST'], y['MIN_CONFIDENCE'], y['NMS_MAX_OVERLAP'], y['MAX_IOU_DISTANCE'],
                    y['MAX_AGE'], y['N_INIT'], y['NN_BUDGET'])

    return opt

