from time import time
import warnings
import os
import subprocess
from DataHelper import ConfigManager
import cv2
import numpy as np
from math import sqrt
import tensorflow as tf
from scipy import interpolate


def getTFsess():
    return tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))


""" Texture Generation """


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


def color_grad_2_pts_x(img, x0, x1, y, left_color, right_color):
    length = x1 - x0
    img[y, x0:x1, 0] = np.fromfunction(
        lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
    img[y, x0:x1, 1] = np.fromfunction(
        lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
    img[y, x0:x1, 2] = np.fromfunction(
        lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
    return img


def color_grad_2_pts_y(img, y0, y1, x, left_color, right_color):
    length = y1 - y0
    img[y0:y1, x, 0] = np.fromfunction(
        lambda _x: left_color[0] * (1 - _x / length) + right_color[0] * (_x / length), (length,))
    img[y0:y1, x, 1] = np.fromfunction(
        lambda _x: left_color[1] * (1 - _x / length) + right_color[1] * (_x / length), (length,))
    img[y0:y1, x, 2] = np.fromfunction(
        lambda _x: left_color[2] * (1 - _x / length) + right_color[2] * (_x / length), (length,))
    return img


def image_expansion_v2(img, internal=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    if internal:
        img[:, :, 2] *= 1.2
        img[:, :, 1] *= 0.8
    avg_color = img[img.sum(-1) > 0].mean(0)
    maskHSV = cv2.inRange(img, avg_color - np.array([10, 40, 25]), avg_color + np.array([10, 100, 50]))
    for i in range(maskHSV.shape[0]):
        t = maskHSV[i].nonzero()[0].flatten()
        if t.size > 1:
            maskHSV[i, t[0]:t[-1]] = 255
    resultHSV = cv2.bitwise_and(img, img, mask=maskHSV)

    new_img_x = resultHSV.copy().astype(np.float32)
    img = resultHSV
    for r in range(img.shape[0]):
        t = np.argwhere(img[r].sum(-1) > 0).flatten()
        if t.size > 0:
            left_edge = np.min(t) + 5
            right_edge = np.max(t) - 5

            while img[r, left_edge].sum(-1) <= 5:
                left_edge -= 1
            while img[r, right_edge].sum(-1) <= 5:
                right_edge += 1
            # left edge
            new_img_x = color_grad_2_pts_x(new_img_x, x0=0, x1=left_edge, y=r,
                                           left_color=img[r, left_edge] * 0.5 + avg_color * 0.5,
                                           right_color=img[r, left_edge])

            # right edge
            new_img_x = color_grad_2_pts_x(new_img_x, x0=right_edge, x1=img.shape[1], y=r,
                                           left_color=img[r, right_edge, :],
                                           right_color=img[r, right_edge, :] * 0.5 + avg_color * 0.5)

            # internal
            if internal:
                left_edge = np.min(t) - 5
                right_edge = np.max(t) + 5
                while img[r, left_edge].sum(-1) <= 5:
                    left_edge += 1
                while img[r, right_edge].sum(-1) <= 5:
                    right_edge -= 1
                new_img_x = color_grad_2_pts_x(new_img_x, x0=left_edge, x1=right_edge, y=r,
                                               left_color=new_img_x[r, left_edge],
                                               right_color=new_img_x[r, right_edge])

    new_img_y = new_img_x.copy().astype(np.float32)
    for c in range(new_img_y.shape[1]):
        t = np.argwhere(new_img_y[:, c, :].sum(-1) > 0).flatten()
        if t.size > 0:
            left_edge = np.min(t) + 5
            right_edge = np.max(t) - 5
            while new_img_y[left_edge, c].sum(-1) <= 5:
                left_edge -= 1
            while new_img_y[right_edge, c].sum(-1) <= 5:
                right_edge += 1
            # left edge
            new_img_y = color_grad_2_pts_y(new_img_y, y0=0, y1=left_edge, x=c,
                                           left_color=new_img_y[left_edge, c] * 0.5 + avg_color * 0.5,
                                           right_color=new_img_y[left_edge, c])
            new_img_x = color_grad_2_pts_y(new_img_x, y0=0, y1=left_edge, x=c,
                                           left_color=new_img_y[left_edge, c] * 0.5 + avg_color * 0.5,
                                           right_color=new_img_y[left_edge, c])

            # right edge
            new_img_y = color_grad_2_pts_y(new_img_y, y0=right_edge, y1=img.shape[0], x=c,
                                           left_color=new_img_y[right_edge, c, :],
                                           right_color=new_img_y[right_edge, c, :] * 0.5 + avg_color * 0.5)
            new_img_x = color_grad_2_pts_y(new_img_x, y0=right_edge, y1=img.shape[0], x=c,
                                           left_color=new_img_y[right_edge, c, :],
                                           right_color=new_img_y[right_edge, c, :] * 0.5 + avg_color * 0.5)
            if internal:
                left_edge = np.min(t) - 5
                right_edge = np.max(t) + 5
                while new_img_y[left_edge, c].sum(-1) <= 5:
                    left_edge += 1
                while new_img_y[right_edge, c].sum(-1) <= 5:
                    right_edge -= 1
                new_img_y = color_grad_2_pts_y(new_img_y, y0=left_edge, y1=right_edge, x=c,
                                               left_color=new_img_y[left_edge, c, :],
                                               right_color=new_img_y[right_edge, c, :] * 0.5 + avg_color * 0.5)
    img_recover = cv2.addWeighted(new_img_x, 0.5, new_img_y, 0.5, 0)
    img_recover = img_recover.round().clip(0, 255).astype(np.uint8)
    img_recover = cv2.cvtColor(img_recover, cv2.COLOR_HSV2BGR)
    return img_recover


# def image_expansion(img, internal=False):
#     # new_img = np.zeros_like(img, dtype=np.float32)
#     new_img = img.copy().astype(np.float32)
#     _, edges_along_y, edges_along_x = edge_detection(img, th=5)
#     color = np.mean(img[np.argwhere(edges_along_y[:, 0] > 0), img.shape[1] // 2, :], axis=0)
#     for _y, (_x0, _x1) in enumerate(edges_along_y):
#         if _x0 != 0 and _x1 != img.shape[1]:
#             # appends edges
#             new_img[_y, -1, :] = new_img[_y, 0, :] = color
#             # left edge
#             length = _x0
#             left_color = new_img[_y, 0, :]
#             right_color = new_img[_y, _x0, :]
#             new_img[_y, :_x0, 0] = np.fromfunction(
#                 lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#             new_img[_y, :_x0, 1] = np.fromfunction(
#                 lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#             new_img[_y, :_x0, 2] = np.fromfunction(
#                 lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#             # right edge
#             length = img.shape[1] - _x1
#             left_color = new_img[_y, _x1, :]
#             right_color = new_img[_y, -1, :]
#             new_img[_y, _x1:, 0] = np.fromfunction(
#                 lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#             new_img[_y, _x1:, 1] = np.fromfunction(
#                 lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#             new_img[_y, _x1:, 2] = np.fromfunction(
#                 lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#             # internal
#             if internal:
#                 length = _x1 - _x0
#                 left_color = new_img[_y, _x0, :]
#                 right_color = new_img[_y, _x1, :]
#                 new_img[_y, _x0:_x1, 0] = np.fromfunction(
#                     lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#                 new_img[_y, _x0:_x1, 1] = np.fromfunction(
#                     lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#                 new_img[_y, _x0:_x1, 2] = np.fromfunction(
#                     lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#     # end of x padding
#     # _, _, edges_along_x = edge_detection(new_img, th=5)
#     color = np.mean(img[img.shape[0] // 2, np.argwhere(edges_along_x[:, 0] > 0), :], axis=0)
#     # for _x, (_y0, _y1) in enumerate(edges_along_x):
#     _y0 = np.argwhere(edges_along_y[:, 0] > 0).min()
#     _y1 = np.argwhere(edges_along_y[:, 0] > 0).max()
#     for _x in range(img.shape[1]):
#         if _y0 != 0 and _y1 != img.shape[0]:
#             # appends edges
#             new_img[0, _x, :] = new_img[-1, _x, :] = color
#             # left edge
#             length = _y0
#             left_color = new_img[0, _x, :]
#             right_color = new_img[_y0, _x, :]
#             new_img[:_y0, _x, 0] = np.fromfunction(
#                 lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#             new_img[:_y0, _x, 1] = np.fromfunction(
#                 lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#             new_img[:_y0, _x, 2] = np.fromfunction(
#                 lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#             # right edge
#             length = img.shape[0] - _y1
#             left_color = new_img[_y1, _x, :]
#             right_color = new_img[-1, _x, :]
#             new_img[_y1:, _x, 0] = np.fromfunction(
#                 lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#             new_img[_y1:, _x, 1] = np.fromfunction(
#                 lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#             new_img[_y1:, _x, 2] = np.fromfunction(
#                 lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#             # # internal
#             # length = _y1 - _y0
#             # left_color = new_img[_y0, _x, :]
#             # right_color = new_img[_y1, _x, :]
#             # new_img[_y0:_y1, _x, 0] = np.fromfunction(
#             #     lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
#             # new_img[_y0:_y1, _x, 1] = np.fromfunction(
#             #     lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
#             # new_img[_y0:_y1, _x, 2] = np.fromfunction(
#             #     lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))
#     # end of y padding
#     new_img = new_img.round().clip(0, 255).astype(np.uint8)
#     return new_img


def genText(img_path, output_path, size=None, internal=False):
    assert size is None or (type(size) == tuple and len(size) == 3)

    img = cv2.imread(img_path)
    if size is None:
        size = img.shape
    new_img = np.zeros(size, dtype=np.uint8)
    new_img[(size[0] * 3 // 5 - img.shape[0] // 2):(size[0] * 3 // 5 + img.shape[0] // 2),
    (size[1] - img.shape[1]) // 2:(size[1] + img.shape[1]) // 2, :] = img
    # img = image_expansion_v2(new_img, internal)
    img = image_expansion_v3(new_img, internal)
    # img = new_img
    cv2.imwrite(output_path, img)
    return


def image_expansion_v3(new_img, internal):
    img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV).astype(np.float64)
    avg_color = img[np.logical_and(img.sum(-1) > 10, img.sum(-1) < 700)].mean(0)
    maskHSV = cv2.inRange(img, avg_color - np.array([10, 40, 40], dtype=np.float64),
                          avg_color + np.array([20, 30, 50], dtype=np.float64))
    for i in range(maskHSV.shape[0]):
        t = maskHSV[i].nonzero()[0].flatten()
        if t.size > 1:
            maskHSV[i, t[0]:t[-1]] = 255
    resultHSV = cv2.bitwise_and(img, img, mask=maskHSV)
    new_img_x = resultHSV.copy().astype(np.float32)
    left_edge = np.zeros(img.shape[0], dtype=np.uint32)
    right_edge = np.full(img.shape[0], img.shape[1], dtype=np.uint32)
    for _y in range(img.shape[0]):
        t = np.argwhere(img[_y].sum(-1) > 0).flatten()
        if t.size > 0:
            k = 4
            left_edge[_y] = np.min(t) + k
            right_edge[_y] = np.max(t) - k
            k = 1
            kind = "slinear"
            x_fit = np.concatenate(([0], np.arange(left_edge[_y], left_edge[_y] + k)), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img_x[_y, left_edge[_y]:left_edge[_y] + k, :]), 0)
            fl = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            x_fit = np.concatenate(([new_img_x.shape[1]], np.arange(right_edge[_y] - k, right_edge[_y])), 0)
            y_fit = np.concatenate((avg_color.reshape(1, 3), new_img_x[_y, right_edge[_y] - k: right_edge[_y], :]), 0)
            fr = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
            new_img_x[_y, :left_edge[_y]] = fl(np.arange(left_edge[_y])).clip(0, 255)
            new_img_x[_y, right_edge[_y]:] = fr(np.arange(right_edge[_y], new_img_x.shape[1])).clip(0, 255)
    for _y in range(img.shape[0]):
        for _x in reversed(range(0, left_edge[_y])):
            new_img_x[_y, _x] = 0.33 * new_img_x[_y - 1, _x] + 0.34 * new_img_x[_y, _x + 1] + 0.33 * new_img_x[
                _y + 1, _x]
        for _x in range(right_edge[_y], new_img_x.shape[1]):
            new_img_x[_y, _x] = 0.33 * new_img_x[_y - 1, _x] + 0.34 * new_img_x[_y, _x - 1] + 0.33 * new_img_x[
                _y + 1, _x]
    # up_edge = np.zeros(img.shape[1], dtype=np.uint32)
    # down_edge = np.full(img.shape[1], img.shape[0], dtype=np.uint32)
    # for _x in range(img.shape[1]):
    #     t = np.argwhere(img[:, _x, :].sum(-1) > 0).flatten()
    #     if t.size > 0:
    #         k = 4
    #         up_edge[_x] = np.min(t) + k
    #         down_edge[_x] = np.max(t) - k
    #         k = 1
    #         kind = "slinear"
    #         x_fit = np.concatenate(([0], np.arange(up_edge[_x], up_edge[_x] + k)), 0)
    #         y_fit = np.concatenate((avg_color.reshape(1, 3), new_img_x[up_edge[_x]:up_edge[_x] + k, _x, :]), 0)
    #         fl = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
    #         x_fit = np.concatenate(([new_img_x.shape[0]], np.arange(down_edge[_x] - k, down_edge[_x])), 0)
    #         y_fit = np.concatenate((avg_color.reshape(1, 3), new_img_x[down_edge[_x] - k: down_edge[_x], _x, :]), 0)
    #         fr = interpolate.interp1d(x_fit, y_fit, kind=kind, axis=0, fill_value="extrapolate")
    #         new_img_x[:up_edge[_x], _x] = fl(np.arange(up_edge[_x])).clip(0, 255)
    #         new_img_x[down_edge[_x]:, _x] = fr(np.arange(down_edge[_x], new_img_x.shape[0])).clip(0, 255)
    # for _x in range(img.shape[1]):
    #     for _y in reversed(range(0, up_edge[_x])):
    #         new_img_x[_y, _x] = 0.33 * new_img_x[_y, _x - 1] + 0.34 * new_img_x[_y + 1, _x] + 0.33 * new_img_x[
    #             _y, _x + 1]
    #     for _y in range(down_edge[_x], new_img_x.shape[0]):
    #         new_img_x[_y, _x] = 0.33 * new_img_x[_y, _x - 1] + 0.34 * new_img_x[_y - 1, _x] + 0.33 * new_img_x[
    #             _y, _x + 1]
    img_recover = new_img_x.round().clip(0, 255).astype(np.uint8)
    img_recover = cv2.cvtColor(img_recover, cv2.COLOR_HSV2BGR)
    return img_recover

"""Call Blender"""


def blender_wrapper(blender_file, script_file_path, input_data, texture, hair, mask, output, gen_hair, background=True):
    # LOAD CONFIG FILE
    configManager = ConfigManager('.\\config.ini')
    keyAndValue = {}
    keyAndValue['INPUT_DATA'] = input_data
    keyAndValue['TEXTURE_DATA'] = texture
    keyAndValue['HAIR_DATA'] = hair
    keyAndValue['MASK_DATA'] = mask
    keyAndValue['OUT_DATA'] = output
    keyAndValue['HAIR'] = gen_hair
    configManager.addPairs(keyAndValue)
    # SAVE CONFIG FILE

    blender = r".\\Blender\\blender.exe"
    if os.path.exists(blender):
        if background:
            cmd = "{} -b {} -P {}".format(blender, blender_file, script_file_path)
        else:
            cmd = "{} {} -P {}".format(blender, blender_file, script_file_path)
        print(cmd)
        try:
            return_code = subprocess.call(cmd.split(' '), shell=True)
            if return_code:
                raise Exception("Unknown Error for blender_wrapper")
        except Exception as e:
            print(e)
            print("CMD:")
            print(cmd.split(' '))
    else:
        print("\tBlender not found")
        print("\tMake a Symbolic Link of Blender root folder BY")
        print("\tWindows: System console")
        print("\t\t cd <PROJECT FOLDER>")
        print("\t\t mklink /D <BLNDER FOLDER> Blender")
        print("\n\tLinux/Mac: bash terminal")
        print("\t\t cd <PROJECT FOLDER>")
        print("\t\t sudo ln -s <BLNDER FOLDER> Blender")
    return


"""Utility"""


def dist(pt1, pt2):
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def display(img, name="Img", time=0, encode="BGR"):
    if type(img) != np.ndarray:
        if type(img) == str:
            img = cv2.imread(img)
        else:
            raise TypeError("Should be img (numpy.ndarray) or img path (str), but {} found".format(type(img)))
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
        encode = "BGR"
    if img.ndim == 3:
        img = img[..., [encode.find('B'), encode.find('G'), encode.find('R')]]  # to BGR
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def clear_all_output():
    pass


def time_it_wrapper(callback, name="", args=(), kwargs={}):
    print(name, ": ")
    start_time = time()
    temp = None
    if callback:
        temp = callback(*args, **kwargs)
    print("\ttime={:.2f}s".format(time() - start_time))
    return temp


def main():
    """
    Main
    :return:
    """
    """Ask for input"""
    # img_path = input("Path of image: ")
    img_path = "0.jpg"
    os.chdir(r"C:\Users\KTL\Desktop\FYP-code\\")
    global_start = time()
    """Import constants from config file"""
    configManager = ConfigManager('.\\config.ini')
    json_data = configManager.getAll()
    OBJ_HEAD_MODEL_HAIR = json_data["OBJ_HEAD_MODEL_HAIR"]
    DIR_INPUT = json_data["DIR_INPUT"]
    DIR_TEXTURE = json_data["DIR_TEXTURE"]
    DIR_HAIR = json_data["DIR_HAIR"]
    DIR_MASK = json_data["DIR_MASK"]
    DIR_OUT = json_data["DIR_OUT"]
    DIR_KPTS = json_data["DIR_KPTS"]
    INPUT_DATA = json_data["INPUT_DATA"]
    TEXTURE_DATA = json_data["TEXTURE_DATA"]
    HAIR_DATA = json_data["HAIR_DATA"]
    MASK_DATA = json_data["MASK_DATA"]
    OUT_DATA = json_data["OUT_DATA"]
    HAIR = json_data["HAIR"]

    INPUT_DATA = json_data['INPUT_DATA'] = img_path
    TEXTURE_DATA = json_data["TEXTURE_DATA"] = img_path
    MASK_DATA = json_data["MASK_DATA"] = "{}.obj".format(img_path[:-4])
    OUT_DATA = json_data["OUT_DATA"] = "{}.obj".format(img_path[:-4])
    configManager.addPairs(json_data)
    assert os.path.exists(os.path.join(DIR_INPUT, img_path))

    """Setup"""
    warnings.filterwarnings("ignore")
    print("Importing packages: ")
    start_time = time()
    from PRNet.myPRNET import genPRMask
    print("\ttime={:.2f}s".format(time() - start_time))

    """END"""
    """Geometry"""
    time_it_wrapper(None, "Generating Geometry")
    """Mask"""
    # time_it_wrapper(genPRMask, "Generating Mask", (os.path.join(DIR_INPUT, img_path), DIR_MASK),
    #                 kwargs={'isMask': False})
    """Texture"""
    time_it_wrapper(genText, "Generating External Texture", (
        os.path.join(DIR_MASK, "{}_texture_2.png".format(MASK_DATA[:-4])), os.path.join(DIR_TEXTURE, TEXTURE_DATA),
        (512, 512, 3), False))
    # time_it_wrapper(genText, "Generating Internal Texture", (
    #     os.path.join(DIR_MASK, "{}_texture.png".format(MASK_DATA[:-4])),) * 2)
    """Alignment"""
    # time_it_wrapper(blender_wrapper, "Alignment",
    #                 args=(".\\new_geometry.blend", ".\\blender_script\\geo.py", INPUT_DATA, TEXTURE_DATA, HAIR_DATA,
    #                       MASK_DATA, OUT_DATA, HAIR, False))
    print("Output to: {}".format(os.path.join(os.getcwd(), DIR_OUT, OUT_DATA)))
    print("Total_time: {:.2f}".format(time() - global_start))
    return


if __name__ == '__main__':
    main()
