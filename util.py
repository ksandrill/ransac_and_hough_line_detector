import numpy as np


def get_dX(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dx = convolve(image, kernel)
    return dx


def get_dY(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    dy = convolve(image, kernel)
    return dy


def normalize_image(x: np.ndarray, a: float, b: float):
    min_val = np.min(x)
    max_val = np.max(x)
    return a + (x - min_val) * (b - a) / (max_val - min_val)


def convolve(image: np.ndarray, kernel: np.ndarray):
    h_kernel, w_kernel = kernel.shape
    h_image, w_image = image.shape
    h_offset = h_kernel // 2
    w_offset = w_kernel // 2
    padded_image = np.zeros((h_image + h_offset * 2, w_image + w_offset * 2))
    padd_h_image = padded_image.shape[0] - h_offset
    padd_w_image = padded_image.shape[1] - w_offset
    padded_image[h_offset:padd_h_image, w_offset:padd_w_image] = image
    output = np.zeros_like(padded_image)
    for i in range(h_offset, h_image):
        for j in range(w_offset, w_image):
            output[i, j] = np.sum(kernel * padded_image[i - h_offset: i + h_offset + 1, j - w_offset: j + w_offset + 1])
    return output[h_offset:padd_h_image, w_offset:padd_w_image]


def create_color_gray(picture: np.ndarray):
    color_gray = np.zeros((picture.shape[0], picture.shape[1], 3))
    for i in range(color_gray.shape[0]):
        for j in range(color_gray.shape[1]):
            color_gray[i][j] = [picture[i][j], picture[i][j], picture[i][j]]
    return color_gray
