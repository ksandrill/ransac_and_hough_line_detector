import cv2
import numpy as np

import util


# rho = x * cos_theta + y * sin_theta
# 0 <= theta <= 180
# -d < rho < d, where d is d is size of image's diagonal


def hough_lines(grad_img: np.ndarray, angle_step: int = 1, edge_thr: int = 255) -> (
        np.ndarray, np.ndarray, np.ndarray, list[list[list[(int, int)]]]):
    h_image, w_image = grad_img.shape[:2]
    diag_image = int(np.sqrt(h_image ** 2 + w_image ** 2))
    thetas = np.deg2rad(np.arange(0, 180, step=angle_step))
    rhos = np.linspace(-diag_image, diag_image, diag_image * 2)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    num_thetas = len(thetas)
    num_rhos = len(rhos)
    accumulator = np.zeros((num_rhos, num_thetas))
    point_list = [[[] for _ in range(num_thetas)] for _ in range(num_rhos)]
    edges_points = np.argwhere(grad_img >= edge_thr)
    for (y, x) in edges_points:
        for theta_idx in range(num_thetas):
            rho = x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx][theta_idx] += 1
            point_list[rho_idx][theta_idx].append((x, y))
    return accumulator, thetas, rhos, point_list


def draw_hough_lines(img: np.ndarray, accumulator: np.ndarray, point_list: list[list[list[(int, int)]]],
                     votes_thr: int = 50) -> None:
    indexes = np.argwhere(accumulator > votes_thr)
    for (rho_idx, theta_idx) in indexes:
        points = point_list[rho_idx][theta_idx]
        cv2.line(img, min(points), max(points), color=255)


def main():
    img = cv2.imread('pictures/glavnaya-doroga-eto-doroga_4.jpg', 0)
    img = img.astype(float) / 255.0
    DX = util.get_dX(img)
    DY = util.get_dY(img)  # Prewitt operator
    grad_image = np.sqrt(DX ** 2 + DY ** 2)
    accumulator, thetas, rhos, point_list = hough_lines(grad_image, edge_thr=0.7)
    draw_hough_lines(img, accumulator, point_list, 65)
    cv2.imshow('edges', img.astype(float))
    cv2.waitKey(0)
    cv2.imwrite('output/hough_out2.png', img)


if __name__ == '__main__':
    main()
