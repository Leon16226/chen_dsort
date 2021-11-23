import numpy as np
from strategy.todo import todo
import cv2
from .shape import intersects, poly_area, Cal_area_2poly


def postprocess_track(nn,
                      opt, im0s, real_box,
                      pools, areas, lock):

    # real_box : [x1, y1, x2, y2, cls, conf]

    # areas
    # 0 : 异常停车
    a_ill_park = np.array(areas[0][nn])

    # test
    pp = a_ill_park[0].reshape((-1, 1, 2))
    cv2.polylines(im0s, [pp], True, (211, 0, 148))

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

    vehicles = (0, 1, 13, 14, 16, 17)
    people_and_novehicles = (8, 10, 12, 15)
    materials = (2, 3, 4, 5, 6, 7)

    my_vehicles = real_box[[v[4] in vehicles for v in real_box]]
    # my_non_vehicles = real_box[real_box[:, 4] in people_and_novehicles]
    # my_materials = real_box[real_box[:, 4] in materials]

    # box
    # 0 :  异常停车
    # 1 ： 行人或非机动车(包括行人、自行车、三轮车、摩托车)
    # 2 ： 抛洒物
    # 3 ： 异常行驶
    # 4 ： 拥堵检测
    b_ill_park = []
    b_people = []
    b_material = []
    b_ill_driving = []
    b_crowed = []
    for i, ve in enumerate(my_vehicles):
        p = np.array([(ve[2] + ve[0])/2, (ve[3] + ve[1])/2])
        # x个异常停车检测区域
        for area in a_ill_park:
            if intersects(p, area):
                b_ill_park.append(ve)

    print('ill_park:', len(b_ill_park))

    c_box = {
             0: np.array(b_ill_park),
             1: np.array(b_people),
             2: np.array(b_material),
             3: np.array(b_ill_driving),
             4: np.array(b_crowed)}

    todo(c_box, pools, opt, im0s, lock)