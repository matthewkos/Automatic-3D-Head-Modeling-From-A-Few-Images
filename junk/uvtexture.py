import cv2
import numpy as np
from time import time
from math import sqrt


# def image_expansion(img, edges_along_y, edges_along_x, mode='xyi'):
def image_expansion(img, mode=""):
    # new_img = np.zeros_like(img, dtype=np.float32)
    new_img = img.copy().astype(np.float32)
    _, edges_along_y, edges_along_x = edge_detection(img, th=5)
    color = np.mean(img[np.argwhere(edges_along_y[:, 0] > 0), img.shape[1] // 2, :], axis=0)
    for _y, (_x0, _x1) in enumerate(edges_along_y):
        if _x0 != 0 and _x1 != img.shape[1]:
            # appends edges
            new_img[_y, -1, :] = new_img[_y, 0, :] = color
            # left edge
            length = _x0
            left_color = new_img[_y, 0, :]
            right_color = new_img[_y, _x0, :]
            new_img[_y, :_x0, 0] = np.fromfunction(
                lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            new_img[_y, :_x0, 1] = np.fromfunction(
                lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            new_img[_y, :_x0, 2] = np.fromfunction(
                lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
            # right edge
            length = img.shape[1]-_x1
            left_color = new_img[_y, _x1, :]
            right_color = new_img[_y, -1, :]
            new_img[_y, _x1:, 0] = np.fromfunction(
                lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            new_img[_y, _x1:, 1] = np.fromfunction(
                lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            new_img[_y, _x1:, 2] = np.fromfunction(
                lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))

            # internal
            if 'i' in mode:
                length = _x1 - _x0
                left_color = new_img[_y, _x0, :]
                right_color = new_img[_y, _x1, :]
                new_img[_y, _x0:_x1, 0] = np.fromfunction(
                    lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
                new_img[_y, _x0:_x1, 1] = np.fromfunction(
                    lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
                new_img[_y, _x0:_x1, 2] = np.fromfunction(
                    lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
    # end of x padding
    # _, _, edges_along_x = edge_detection(new_img, th=5)
    color = np.mean(img[img.shape[0] // 2, np.argwhere(edges_along_x[:, 0] > 0), :], axis=0)
    # for _x, (_y0, _y1) in enumerate(edges_along_x):
    _y0 = np.argwhere(edges_along_y[:, 0] > 0).min()
    _y1 =np.argwhere(edges_along_y[:, 0] > 0).max()
    for _x in range(img.shape[0]):
        if _y0 != 0 and _y1 != img.shape[0]:
            # appends edges
            new_img[0, _x, :] = new_img[-1, _x, :] = color
            # left edge
            length = _y0
            left_color = new_img[0, _x, :]
            right_color = new_img[_y0, _x, :]
            new_img[:_y0, _x, 0] = np.fromfunction(
                lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            new_img[:_y0, _x, 1] = np.fromfunction(
                lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            new_img[:_y0, _x, 2] = np.fromfunction(
                lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
            # right edge
            length = img.shape[0]-_y1
            left_color = new_img[_y1, _x, :]
            right_color = new_img[-1, _x, :]
            new_img[_y1:, _x, 0] = np.fromfunction(
                lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            new_img[_y1:, _x, 1] = np.fromfunction(
                lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            new_img[_y1:, _x, 2] = np.fromfunction(
                lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))

            # # internal
            # length = _y1 - _y0
            # left_color = new_img[_y0, _x, :]
            # right_color = new_img[_y1, _x, :]
            # new_img[_y0:_y1, _x, 0] = np.fromfunction(
            #     lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            # new_img[_y0:_y1, _x, 1] = np.fromfunction(
            #     lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            # new_img[_y0:_y1, _x, 2] = np.fromfunction(
            #     lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
    # end of y padding
    new_img = new_img.round().clip(0, 255).astype(np.uint8)
    return new_img


def dist(pt1, pt2):
    # return np.linalg.norm(pt2-pt1)
    # return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def display(img, name="Img", time=0, encode="BGR"):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
        encode = "BGR"
    if img.ndim == 3:
        img = img[..., [encode.find('B'), encode.find('G'), encode.find('R')]]  # to BGR
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def edge_detection(img, th=10):
    """
    detect the edge
    :param img: source image
    :param th: threshold to decide what is black
    :return: grad, edge_along_y, edge_along_x
    """
    src = img.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.convertScaleAbs(
        cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
    grad_y = cv2.convertScaleAbs(
        cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    # display(grad, "output")

    edge_along_y = np.zeros((img.shape[0], 2), dtype=np.uint16)
    edge_along_x = np.zeros((img.shape[1], 2), dtype=np.uint16)
    for _y in range(edge_along_y.shape[0]):
        temp = np.argwhere(grad_x[_y, :] > th)
        if np.any(temp):
            edge_along_y[_y, 0] = np.min(temp) + 1
            edge_along_y[_y, 1] = np.max(temp) - 1
    for _x in range(edge_along_x.shape[0]):
        temp = np.argwhere(grad_x[:, _x] > th)
        if np.any(temp):
            edge_along_x[_x, 0] = np.min(temp) + 1
            edge_along_x[_x, 1] = np.max(temp) - 1
    return grad, edge_along_y, edge_along_x



if __name__ == '__main__':
    start_time = time()
    img = cv2.imread(r'Data\mask\0_texture.png')
    # edge_img, edge_along_y, edge_along_x = edge_detection(img, th=5)
    new_img = image_expansion(img,'i')
    display(new_img, "v4")
