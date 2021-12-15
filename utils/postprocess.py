import numpy as np
from strategy.todo import todo
import cv2
from .shape import intersects, poly_area, Cal_area_2poly


index_of_vehicles = (0, 1, 13, 14, 16, 17)
index_of_people = (8, 10, 12, 15)
index_of_materials = (2, 3, 4, 5, 6, 7)


def postprocess_track(url, point, dict_strategy,
                      height, width, labels,
                      im0s, real_box,
                      point_detect_areas,
                      matrix_park, matrixs_crowd,
                      matrix_service):

    # real_box : [x1, y1, x2, y2, cls, conf]
    # 此时box是相对坐标

    # areas
    # 0. 异常停车
    # 1. 行人
    # 2. 抛撒物
    # 3. 异常行驶

    area_of_park = point_detect_areas['IllegalPark']
    area_of_crowd = point_detect_areas['Crowed']

    # 0:car 轿车
    # 1:truck 卡车
    # 2:cup
    # 3:cans
    # 4:bottle
    # 5:mealbox
    # 6:box
    # 7：bag
    # 8：person ----人------
    # 9: barricade
    # 10: motorbike ----摩托车-----
    # 11: bullbarrels
    # 12: threebicycle -----三轮车-------
    # 13: bus 汽车
    # 14: tanker 油罐车
    # 15: bicycle --------自行车---------
    # 16: tzc 特种车
    # 17: trailer 拖车
    # 18: fomabox
    # 19: fire

    detected_park = []
    detected_people = []
    detected_materials = []
    detected_driving = []
    detected_crowed = []

    for box in real_box:
        # 预测框的中心坐标
        center = np.array([(box[2] + box[0]) / 2 * width, (box[3] + box[1]) / 2 * height])

        if box[4] in index_of_vehicles:
            # 是否在异常停车区域？
            if matrix_service[0] == 1:
                for area in area_of_park:
                    if intersects(center, area):
                        detected_park.append(box)
            # 是否在拥堵检测区域？
            if matrixs_crowd[4] == 1:
                for area in area_of_crowd:
                    if intersects(center, area):
                        detected_park.append(box)
        elif box[4] in index_of_people:
            detected_people.append(box)
        elif box[4] in index_of_materials:
            detected_materials.append(box)

    print('rtsp:', 'the number of boxes in ill_park area:', len(detected_park))
    print('rtsp:', 'the number of boxes in people area:', len(detected_people))

    boxes_of_different_areas = {
             0: np.array(detected_park),
             1: np.array(detected_people),
             2: np.array(detected_materials),
             3: np.array(detected_driving),
             4: np.array(detected_crowed)}

    todo(url, point, boxes_of_different_areas, dict_strategy,
         im0s, labels, height, width,
         matrix_park, matrixs_crowd)