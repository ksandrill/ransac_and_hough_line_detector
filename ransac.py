from typing import Callable, Optional

import cv2
import numpy as np

# Ax + By + C = 0
# A = y1 -y2
# B = x2 -  x1
# C = x1y1  - x2y2
# (x1,y1) and (x2,y2) in Ax+ By + C = 0
# dist = |Axp + B * yp + C|/ sqrt(A^2 + B^2)
import util


def line_ransac_model(data_points: np.ndarray, dist_tolerance: float) -> tuple[list[(float, float)], np.ndarray]:
    data_points_count = len(data_points)
    if data_points_count < 2:
        return [], np.array([])
    idx1 = np.random.randint(0, data_points_count)
    idx2 = np.random.randint(0, data_points_count)
    p1, p2 = data_points[idx1], data_points[idx2]
    x1, y1 = p1
    x2, y2 = p2
    model = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])  # a,b,c
    norm = np.sqrt(model[0] ** 2 + model[1] ** 2)
    inliers = []
    for (x, y) in data_points:
        dist = np.abs(model.dot(np.array([x, y, 1]))) / norm
        if dist <= dist_tolerance:
            inliers.append((x, y))
    return inliers, model


def ransac(data_points: np.ndarray, dist_tolerance: float, iterations: int,
           model: Callable[[np.ndarray, float], tuple[list[(float, float)], np.ndarray]]) -> tuple[
    list[(float, float)], np.ndarray]:
    result_inliers, result_model = model(data_points, dist_tolerance)
    for iteration in range(iterations - 1):
        cur_inliers, cur_model = model(data_points, dist_tolerance)
        if len(cur_inliers) >= len(result_inliers):
            result_inliers = cur_inliers
            result_model = cur_model
    return result_inliers, result_model


# def test():
#     img = np.zeros((200, 200))
#     h, w = img.shape
#     x_val = np.random.randint(0, 200, size=4)
#     y_val = np.random.randint(0, 200, size=3)
#     for y in y_val:
#         for x in x_val:
#             img[y][x] = 255
#
#     for i in range(0, h, 5):
#         for j in range(0, w, 5):
#             if i == j:
#                 img[i, i] = 255
#     edges = np.argwhere(img == 255)
#     result_points, result_model = ransac(edges, 4, iterations=3, model=line_ransac_model)
#     print(result_model)
#     cv2.line(img, pt1=min(result_points), pt2=max(result_points), color=255)
#     cv2.imshow('dafaq', img.astype(np.uint8))
#     cv2.waitKey(0)


def main():
    img = cv2.imread('pictures/glavnaya-doroga-eto-doroga_4.jpg', 0)
    img = img.astype(float) / 255.0
    DX = util.get_dX(img)
    DY = util.get_dY(img)  # Prewitt operator
    grad_image = np.sqrt(DX ** 2 + DY ** 2)
    edges_points = np.argwhere(grad_image >= 0.7)
    for i in range(len(edges_points)):
        edges_points[i][0], edges_points[i][1] = edges_points[i][1], edges_points[i][0]
    data_points = edges_points
    ransac_iteration_number = 30
    for ransac_iter in range(ransac_iteration_number):
        print("iteration: ", ransac_iter + 1)
        points, model = ransac(data_points, dist_tolerance=5, iterations=200, model=line_ransac_model)
        if points:
            cv2.line(img, pt1=min(points), pt2=max(points), color=255)
            set_data_points = set((point[0], point[1]) for point in data_points)
            set_points = set((point[0], point[1]) for point in points)
            data_points = np.array(list(set_data_points - set_points))
            print('model: ', model)
            print('points number: ', len(points))
    cv2.imshow('edges', img.astype(float))
    cv2.waitKey(0)
    cv2.imwrite('output/ransac_out2.png', img)


if __name__ == '__main__':
    main()
