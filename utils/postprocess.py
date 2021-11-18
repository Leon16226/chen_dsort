import numpy as np
from strategy.todo import todo
import cv2
from .shape import intersects, poly_area, Cal_area_2poly


def postprocess_track(outputs,
                      car_id_pool, people_id_pool,
                      material_id_pool, illdri_id_pool,
                      crowed_id_pool,
                      opt, im0s,
                      lock,
                      point2,
                      point3,
                      crows,
                      p_crowed_time,
                      crowed_block,
                      car_num):

    # 硬路肩异常停车区域 蓝色
    # point_s = point2.reshape((-1, 1, 2))
    # cv2.polylines(im0s, [point_s], True, (255, 255, 0))

    # 拥堵检测区域 紫色
    point_c = point3.reshape((-1, 1, 2))
    cv2.polylines(im0s, [point_c], True, (211, 0, 148))

    # box : [x1, y1, x2, y2, id, cls]
    in_track_box = np.array(outputs)



    # 0 : 异常停车
    # 1 ： 行人或非机动车(包括行人、自行车、三轮车、摩托车)
    # 2 ： 抛洒物
    # 3 ： 异常行驶

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

    vehicles = in_track_box[(in_track_box[:, 5] == 0) + (in_track_box[:, 5] == 1) + (in_track_box[:, 5] == 13)
                            + (in_track_box[:, 5] == 14) + (in_track_box[:, 5] == 16) + (in_track_box[:, 5] == 17)]

    people_or_novehicles = in_track_box[(in_track_box[:, 5] == 8) + (in_track_box[:, 5] == 10)
                             + (in_track_box[:, 5] == 12) + (in_track_box[:, 5] == 15)]

    materials = in_track_box[(in_track_box[:, 5] == 2) + (in_track_box[:, 5] == 3) + (in_track_box[:, 5] == 4)
                            + (in_track_box[:, 5] == 5) + (in_track_box[:, 5] == 6) + (in_track_box[:, 5] == 7)]

    illdris = []  # 硬路肩
    crowed = []   # 拥堵区域
    for i, ve in enumerate(vehicles):
        p = np.array([(ve[2] + ve[0])/2, (ve[3] + ve[1])/2])
        if intersects(p, point2):
            illdris.append(ve)
            print("硬路肩有车辆通过")
        elif intersects(p, point3):
            print("拥堵检测区域有车辆通过")
            crowed.append(ve)
    illdris = np.array(illdris)
    crowed = np.array(crowed)

    if not crowed_block[0]:
        # 计算空间占有率
        all_area = poly_area(point3)
        boxs_area = 0
        t_pass = False
        for i, box in enumerate(crows):
            # 要计算相交面积
            points = [box[0], box[1], box[2], box[1],
                    box[2], box[3], box[0], box[3]]
            points = np.array(points).reshape([4, 2])
            boxs_area += Cal_area_2poly(points, point3)
            # 小框位置是否有车辆经过
            p = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
            if intersects(p, p_crowed_time) and not t_pass:
                t_pass = True

        space_rate = 1.0 * boxs_area / all_area
        space_rate = space_rate if space_rate > 0.10 else 0.10
        space_rate = space_rate if space_rate < 1 else 0.95
        print("^^^^^^^^^^^^^^^^^ 空间占有率 ^^^^^^^^^^^^^^^^^^：", space_rate)

        # 计算车辆跟踪率
        if car_num > 4:
            car_o_n = car_num
            car_t_n = vehicles.shape[0] if vehicles.size > 0 else 0
            car_track_rate = 1.0 * car_t_n / car_o_n if (car_o_n != 0) and (car_t_n != 0) else 0.10
            car_track_rate = car_track_rate if car_track_rate < 0.95 else 0.95
        else:
            car_track_rate = 0.10
        print("^^^^^^^^^^^^^^^^^^ 车辆跟踪率 ^^^^^^^^^^^^^^^^^^:", car_track_rate)

        crowed = [crowed, space_rate, t_pass, car_track_rate, crowed_block]
    else:
        print("^^^^^^^^^^^^^^^^^^ 拥堵检测锁定状态 ^^^^^^^^^^^^^^^^^^")

    c_box = {
             0: vehicles,
             1: np.concatenate((people_or_novehicles, vehicles), axis=0),
             2: materials,
             3: illdris,
             4: crowed}

    pool = [car_id_pool, people_id_pool, material_id_pool, illdri_id_pool, crowed_id_pool]
    todo(c_box, pool, opt, im0s, lock)