import shapely.geometry
from shapely.geometry import Polygon


# 两个多变形的相交面积
def Cal_area_2poly(point1, point2):
    poly1 = Polygon(point1).convex_hull
    poly2 = Polygon(point2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


# 判断box中心是否在区域内
def intersects(point, ploy):
    point = shapely.geometry.Point(point)
    poly = Polygon(ploy).convex_hull

    return poly.intersects(point)


# 多边形面积
def poly_area(coords):
    return Polygon(coords).area
