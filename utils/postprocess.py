import numpy as np
from strategy.todo import todo
import cv2
from .shape import intersects, poly_area, Cal_area_2poly


def postprocess_track(nn, point, points,
                      opt, im0s, real_box,
                      pools, areas, lock,
                      matrixs_park):

    # real_box : [x1, y1, x2, y2, cls, conf]

    # areas
    # 0. 异常停车
    # 1. 行人
    # 2. 抛撒物
    # 3. 异常行驶

    a_ill_park = np.array(areas['IllegalPark'][nn]) if 'IllegalPark' in points[nn] else np.array([])
    print('点位：', nn)
    print('区域：', a_ill_park)
    #a_people = np.array(areas['People'][nn]) if str(1) in points[nn] else np.array([])
    #a_material = np.array(areas[2][nn])

    # test
    #if len(a_material) > 0:
    #    pp = a_material[0].reshape((-1, 1, 2))
    #    cv2.polylines(im0s, [pp], True, (211, 0, 148))

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
    my_people = real_box[[p[4] in people_and_novehicles for p in real_box]]
    my_materials = real_box[[m[4] in materials for m in real_box]]

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

    # vehicles
    for i, ve in enumerate(my_vehicles):
        p = np.array([(ve[2] + ve[0])/2, (ve[3] + ve[1])/2])
        # x个异常停车检测区域
        for area in a_ill_park:
            if intersects(p, area):
                b_ill_park.append(ve)

    # # people
    # for i, pe in enumerate(my_people):
    #     p = np.array([(pe[2] + pe[0]) / 2, (pe[3] + pe[1]) / 2])
    #     for area in a_people:
    #         if intersects(p, area):
    #             b_people.append(pe)
    # # 行人还要加车
    b_people_and_car = []
    # b_people.append(b_people)
    # b_people_and_car.append(my_vehicles)

    # materials
    # for i, ma in enumerate(my_materials):
    #     p = np.array([(ma[2] + ma[0]) / 2, (ma[3] + ma[1]) / 2])
    #     for area in a_material:
    #         if intersects(p, area):
    #             b_people.append(ma)

    print('rtsp:', nn, 'the number of ill_park boxes:', len(b_ill_park))
    print('rtsp:', nn, 'the number of people boxes:', len(b_people))

    c_box = {
             0: np.array(b_ill_park),
             1: np.array(b_people_and_car),
             2: np.array(b_material),
             3: np.array(b_ill_driving),
             4: np.array(b_crowed)}

    todo(nn, point, c_box, pools, opt, im0s, lock, matrixs_park[nn],)