import cv2
import numpy as np
from time import time
from math import sqrt
from scipy import interpolate




def hsv2bgr(img):
    return cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def display(img, name="Img", time=0, encode="BGR"):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
        encode = "BGR"
    if img.ndim == 3:
        img = img[..., [encode.find('B'), encode.find('G'), encode.find('R')]]  # to BGR
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def m7():
    img_BGR = cv2.imread(r'Data/mask/KTL_texture_2.png')
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV).astype(np.float64)
    avg_color = img_HSV[np.logical_and(img_HSV.sum(-1) > 10, img_HSV.sum(-1) < 700)].mean(0)
    maskHSV = cv2.inRange(img_HSV, avg_color - np.array([10, 35, 35], dtype=np.float64),
                          avg_color + np.array([10, 25, 35], dtype=np.float64))
    for i in range(maskHSV.shape[0]):
        t = maskHSV[i].nonzero()[0].flatten()
        if t.size > 1:
            maskHSV[i, t[0]:t[-1]] = 255
    masked_HSV = cv2.bitwise_and(img_HSV, img_HSV, mask=maskHSV)
    # set img
    new_img = masked_HSV.copy().astype(np.float32)
    left_edge = np.zeros(masked_HSV.shape[0], dtype=np.uint32)
    right_edge = np.full(masked_HSV.shape[0], img_HSV.shape[1], dtype=np.uint32)
    for _y in range(masked_HSV.shape[0]):
        t = np.argwhere(masked_HSV[_y].sum(-1) > 0).flatten()
        if t.size > 0:
            k = 4
            left_edge[_y] = np.min(t) + k
            right_edge[_y] = np.max(t) - k
            kind = "slinear"
            x_fit = np.concatenate(([left_edge[_y] // 2], np.arange(left_edge[_y], left_edge[_y] + k)), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img[_y, left_edge[_y]:left_edge[_y] + k, :]), 0)
            fl = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            x_fit = np.concatenate(
                ([(new_img.shape[1] + right_edge[_y]) // 2], np.arange(right_edge[_y] - k, right_edge[_y])), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img[_y, right_edge[_y] - k: right_edge[_y], :]), 0)
            fr = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            new_img[_y, left_edge[_y] // 2:left_edge[_y], :] = fl(np.arange(left_edge[_y] // 2, left_edge[_y])).clip(0,
                                                                                                                     255)
            new_img[_y, right_edge[_y]:(new_img.shape[1] + right_edge[_y]) // 2, :] = fr(
                np.arange(right_edge[_y], (new_img.shape[1] + right_edge[_y]) // 2)).clip(0, 255)
            new_img[_y, :left_edge[_y] // 2, :] = avg_color
            new_img[_y, (new_img.shape[1] + right_edge[_y]) // 2:, :] = avg_color
    for _y in range(new_img.shape[0] - 1):
        for _x in reversed(range(0, left_edge[_y])):
            new_img[_y, _x] = 0.33 * new_img[_y - 1, _x] + 0.34 * new_img[_y, _x + 1] + 0.33 * new_img[
                _y + 1, _x]
        for _x in range(right_edge[_y], new_img.shape[1]):
            new_img[_y, _x] = 0.33 * new_img[_y - 1, _x] + 0.34 * new_img[_y, _x - 1] + 0.33 * new_img[
                _y + 1, _x]
    up_edge = np.zeros(img_HSV.shape[1], dtype=np.uint32)
    down_edge = np.full(img_HSV.shape[1], img_HSV.shape[0], dtype=np.uint32)
    for _x in range(img_HSV.shape[1]):
        t = np.argwhere(new_img[:, _x, :].sum(-1) > 0).flatten()
        if t.size > 0:
            k = 4
            up_edge[_x] = np.min(t) + k
            down_edge[_x] = np.max(t) - k
            k = 1
            kind = "slinear"
            x_fit = np.concatenate(([up_edge[_x] // 2], np.arange(up_edge[_x], up_edge[_x] + k)), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img[up_edge[_x]:up_edge[_x] + k, _x, :]), 0)
            fl = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            x_fit = np.concatenate(
                ([(new_img.shape[1] + down_edge[_x]) // 2], np.arange(down_edge[_x] - k, down_edge[_x])), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img[down_edge[_x] - k: down_edge[_x], _x, :]), 0)
            fr = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            new_img[up_edge[_x] // 2:up_edge[_x], _x, :] = fl(np.arange(up_edge[_x] // 2, up_edge[_x])).clip(0, 255)
            new_img[down_edge[_x]:(new_img.shape[0] + down_edge[_y]) // 2, _x, :] = fr(
                np.arange(down_edge[_x], (new_img.shape[0] + down_edge[_y]) // 2)).clip(0, 255)
            new_img[:up_edge[_x] // 2, _x, :] = avg_color
            new_img[(new_img.shape[0] + down_edge[_x]) // 2:, _x, :] = avg_color
    for _x in range(new_img.shape[1] - 1):
        for _y in reversed(range(0, up_edge[_x])):
            new_img[_y, _x] = 0.33 * new_img[_y, _x - 1] + 0.34 * new_img[_y + 1, _x] + 0.33 * new_img[
                _y, _x + 1]
        for _y in range(down_edge[_x], new_img.shape[0]):
            new_img[_y, _x] = 0.33 * new_img[_y, _x - 1] + 0.34 * new_img[_y - 1, _x] + 0.33 * new_img[
                _y, _x + 1]
    out_img = new_img.round().clip(0, 255).astype(np.uint8)
    out_img_BGR = hsv2bgr(out_img)
    display(np.concatenate((img_BGR, hsv2bgr(masked_HSV), out_img_BGR), axis=1))
    return


if __name__ == "__main__":
    m7()
