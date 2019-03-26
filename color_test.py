import cv2
import numpy as np
from time import time
from math import sqrt


def color_gradient_v4(img, edges_x):
    # new_img = np.zeros_like(img, dtype=np.float32)
    new_img = img.copy().astype(np.float32)
    color = np.mean(img[np.argwhere(edges_x[:, 0] > 0), img.shape[1] // 2, :], axis=0)
    for _y, (_x0, _x1) in enumerate(edges_x):
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
            length = _x1 - _x0
            left_color = new_img[_y, _x0, :]
            right_color = new_img[_y, _x1, :]
            new_img[_y, _x0:_x1, 0] = np.fromfunction(
                lambda x: left_color[0] * (1 - x / length) + right_color[0] * (x / length), (length,))
            new_img[_y, _x0:_x1, 1] = np.fromfunction(
                lambda x: left_color[1] * (1 - x / length) + right_color[1] * (x / length), (length,))
            new_img[_y, _x0:_x1, 2] = np.fromfunction(
                lambda x: left_color[2] * (1 - x / length) + right_color[2] * (x / length), (length,))

    new_img = new_img.round().clip(0, 255).astype(np.uint8)
    return new_img


def color_gradient_v3(img, pts, x_range=None, y_range=None, x_loc=None, y_loc=None):
    packed = []
    for i, (_y, _x) in enumerate(pts):
        color = img[_y, _x]
        loc = (_y, _x)
        scale = np.max([dist((_y, _x), (0, 0)),
                        dist((_y, _x), (img.shape[0] - 1, 0)),
                        dist((_y, _x), (0, img.shape[1] - 1)),
                        dist((_y, _x), (img.shape[0] - 1, img.shape[1] - 1))])
        packed.append((color, loc, scale))

    new_img = np.zeros(img.shape)
    if y_range is not None:  # y_range not None
        for _y in y_range:
            if x_range is not None:
                for _x in x_range:
                    for color, loc, scale in packed:
                        new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            elif x_loc is not None:
                _x = x_loc
                for color, loc, scale in packed:
                    new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            else:  # y_range not None, x_range and x_loc are not provided
                raise ValueError("Values are None")
    elif x_range is not None:  # y_range not None
        for _x in x_range:
            if y_loc is not None:
                _y = y_loc
                for color, loc, scale in packed:
                    new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            else:  # x_range not None, y_range and y_loc not provided
                raise ValueError("Values are None")
    else:  # x_range y_range are None
        if x_loc is not None and y_loc is not None:
            _y, _x = y_loc, x_loc
            for color, loc, scale in packed:
                new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
        else:
            raise ValueError("Values are None")
    np.clip(new_img, 0, 255, new_img)
    return new_img.round().astype(np.uint8)


def color_gradient_v2(img, pts, x_range=None, y_range=None, x_loc=None, y_loc=None, mode="a"):
    packed = []
    if mode == 'r':
        dists = np.zeros((len(pts),) * 2)
        for i in range(len(pts)):
            for j in range(len(pts)):
                dists[i, j] = dist(pts[i], pts[j])
        for i, (_y, _x) in enumerate(pts):
            color = img[_y, _x]
            loc = (_y, _x)
            scale = np.max(dists[i])
            packed.append((color, loc, scale))
    elif mode == 'a':
        for i, (_y, _x) in enumerate(pts):
            color = img[_y, _x]
            loc = (_y, _x)
            scale = np.max([dist((_y, _x), (0, 0)),
                            dist((_y, _x), (img.shape[0] - 1, 0)),
                            dist((_y, _x), (0, img.shape[1] - 1)),
                            dist((_y, _x), (img.shape[0] - 1, img.shape[1] - 1))])
            packed.append((color, loc, scale))
    else:
        raise ValueError("Wrong Mode, should be, 'r': relative; 'a': absolute")
    new_img = np.zeros(img.shape)
    if y_range is not None:  # y_range not None
        for _y in y_range:
            if x_range is not None:
                for _x in x_range:
                    for color, loc, scale in packed:
                        new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            elif x_loc is not None:
                _x = x_loc
                for color, loc, scale in packed:
                    new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            else:  # y_range not None, x_range and x_loc are not provided
                raise ValueError("Values are None")
    elif x_range is not None:  # y_range not None
        for _x in x_range:
            if y_loc is not None:
                _y = y_loc
                for color, loc, scale in packed:
                    new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
            else:  # x_range not None, y_range and y_loc not provided
                raise ValueError("Values are None")
    else:  # x_range y_range are None
        if x_loc is not None and y_loc is not None:
            _y, _x = y_loc, x_loc
            for color, loc, scale in packed:
                new_img[_y, _x, :] += color * (1 - dist(loc, (_y, _x)) / scale)
        else:
            raise ValueError("Values are None")
    np.clip(new_img, 0, 255, new_img)
    return new_img.round().astype(np.uint8)


def color_gradient(img, pts, mode="r"):
    packed = []
    if mode == 'r':
        dists = np.zeros((len(pts),) * 2)
        for i in range(len(pts)):
            for j in range(len(pts)):
                dists[i, j] = dist(pts[i], pts[j])
        for i, (y, x) in enumerate(pts):
            color = img[y, x]
            loc = (y, x)
            scale = np.max(dists[i])
            packed.append((color, loc, scale))
    elif mode == 'a':
        for i, (y, x) in enumerate(pts):
            color = img[y, x]
            loc = (y, x)
            scale = np.max([dist((y, x), (0, 0)),
                            dist((y, x), (img.shape[0] - 1, 0)),
                            dist((y, x), (0, img.shape[1] - 1)),
                            dist((y, x), (img.shape[0] - 1, img.shape[1] - 1))])
            packed.append((color, loc, scale))
    else:
        raise ValueError("Wrong Mode, should be, 'r': relative; 'a': absolute")
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for color, loc, scale in packed:
                new_img[i, j, :] += color * (1 - dist(loc, (i, j)) / scale)
    np.clip(new_img, 0, 255, new_img)
    return new_img.round().astype(np.uint8)


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


def color_gradient_demo1():
    MAP_SIZE = (512, 512, 3)  # y,x,c
    initial_color = [{'loc': (MAP_SIZE[0] // 2, 0), 'color': (200, 190, 120)},
                     {'loc': (MAP_SIZE[0] // 2, MAP_SIZE[1] - 1), 'color': (191, 90, 17)},
                     {'loc': (MAP_SIZE[0] // 2, MAP_SIZE[0] // 2), 'color': (255, 255, 255)}]

    img = np.zeros(MAP_SIZE, dtype=np.uint8)
    if len(initial_color) > 1:
        for initial in initial_color:
            l, c = initial.values()
            img[l[0], l[1], :] = np.array(c)
    list_ref_pts = [init_c['loc'] for init_c in initial_color]
    display(img, encode="RGB")
    # constructed an img with initialized color
    # contructed new img
    new_img = color_gradient_v2(img, list_ref_pts, x_range=range(0, img.shape[1]), y_range=range(0, img.shape[0]))
    display(new_img, encode="RGB")
    return


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
    display(grad, "output")

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


def is_black(color: np.array, th=10):
    return np.sum(color) <= th


def find_ref_pts(img, pt, k=3, d=2, valid=is_black):
    """
    find the reference point of the image where it is not black
    :param img: the img
    :param pt: the original point
    :param k: number of points
    :param d: distance
    :param valid: check for ref pt validity
    :return: a list of pt
    """
    pts = []
    m = 1
    while len(pts) < k:
        x_pts = [pt[1] - m * d] * 3 + [pt[1]] * 3 + [pt[1] + m * d] * 3
        y_pts = [pt[0] - m * d, pt[0], pt[0] + m * d] * 3
        for _x, _y in zip(x_pts, y_pts):
            print(_x, _y)
            if 0 <= _x < img.shape[1] and 0 <= _y < img.shape[0] and (_y, _x) not in pts:
                if valid(img[_y, _x, :]):
                    pts.append((_y, _x))
    return pts


def make_img(color, loc, scale, size=(256, 256)):
    new_img = np.zeros((size[0], size[1], 3), dtype=np.float32)
    new_img[:, :, 0] = np.fromfunction(
        lambda i, j: color[0] * (1 - (np.sqrt((i - loc[0]) ** 2 + (j - loc[1]) ** 2)) / scale), size)
    new_img[:, :, 1] = np.fromfunction(
        lambda i, j: color[1] * (1 - (np.sqrt((i - loc[0]) ** 2 + (j - loc[1]) ** 2)) / scale), size)
    new_img[:, :, 2] = np.fromfunction(
        lambda i, j: color[2] * (1 - (np.sqrt((i - loc[0]) ** 2 + (j - loc[1]) ** 2)) / scale), size)
    # new_img = np.transpose(np.stack((img0, img1, img2)), (1, 2, 0))
    return new_img


if __name__ == "__main__":
    # detect edge
    # start_time = time()
    # img = cv2.imread(r'Data\mask\0_texture.png')
    # edge_img, edge_along_y, edge_along_x = edge_detection(img)
    #
    # img_new = img.copy()
    # list_ref_pts = []
    # for _y, edge in enumerate(edge_along_y):
    #     if edge[0] != 0 and edge[1] != 0:
    #         list_ref_pts.append((_y, edge[0]))
    #         if edge[0] < edge[1]:
    #             list_ref_pts.append((_y, edge[1]))
    # for _x, edge in enumerate(edge_along_x):
    #     if edge[0] != 0 and edge[1] != 0:
    #         list_ref_pts.append((edge[0], _x))
    #         if edge[0] < edge[1]:
    #             list_ref_pts.append((edge[1],_x))
    # list_ref_pts = list(set(list_ref_pts))
    # new_img = color_gradient_v2(img, list_ref_pts, x_range=range(0, img.shape[1]), y_range=range(0, img.shape[0]))
    # display(new_img, encode="RGB")
    #
    # # color_gradient_demo1()
    # print("Time Elapsed {:.2f}".format(time()-start_time))

    # for _y in range(img_new.shape[0]):
    #     # if there is a left edge
    #     if edge_x[_y, 0] != 0:
    #         list_of_pts = find_ref_pts(img, (_y, edge_x[_y, 0]),1)
    #         # fill image with x 0 to edge -1, with y = y_ref
    #         img_new = color_gradient_v2(img, pts=list_of_pts, x_range=range(0, edge_x[_y, 0]), y_loc=_y, mode='r')
    #     # if there is a right edge
    #     if edge_x[i, 1] != 0:
    #         pass
    #         # img_new[i, edge_x[i, 1]:, :] = img[i, edge_x[i, 1], :]
    #         # list_of_pts = find_ref_pts(img, (i, edge_x[i, 1]))
    #         # img_new = color_gradient_v2(img, list_of_pts, x=range(edge_x[i, 1], img_new.shape[0]), y_ref=i, mode='r')
    # # img_new = cv2.blur(img_new, (1, 3))
    #
    # # img_new = img.copy()
    # # edge_x = np.zeros((img.shape[0], 2), dtype=np.uint16)
    # # th = 0
    # # for i in range(edge_x.shape[0]):
    # #     temp = np.argwhere(grad_x[i] > th)
    # #     if np.any(temp):
    # #         edge_x[i, 0] = np.min(temp) + 1
    # #         edge_x[i, 1] = np.max(temp) - 1
    # # for i in range(img_new.shape[0]):
    # #     if edge_x[i, 0]:
    # #         img_new[i, :edge_x[i, 0], :] = img[i, edge_x[i, 0], :]
    # #     if edge_x[i, 1]:
    # #         img_new[i, edge_x[i, 1]:, :] = img[i, edge_x[i, 1], :]
    # # img_new = cv2.blur(img_new, (1, 3))
    # # edge_y = np.zeros((img.shape[1], 2), dtype=np.uint16)
    # # th = 0
    # # for i in range(img_new.shape[1]):
    # #     temp = np.argwhere(np.any(img_new[:, i] - th, axis=-1))
    # #     if np.any(temp):
    # #         edge_y[i, 0] = np.min(temp) + 2
    # #         edge_y[i, 1] = np.max(temp) - 2
    # # for i in range(img_new.shape[1]):
    # #     if edge_y[i, 0]:
    # #         img_new[:edge_y[i, 0], i, :] = img_new[edge_y[i, 0], i, :]
    # #     if edge_y[i, 1]:
    # #         img_new[edge_y[i, 1]:, i, :] = img_new[edge_y[i, 1], i, :]
    # # img_new = cv2.blur(img_new, (1, 3))
    # display(img_new, "new_img")

    # color_gradient_demo1()
    # MAP_SIZE=(256,256)
    # initial_color = [{'loc': (MAP_SIZE[0] // 2, 0), 'color': (200, 190, 120), 'scale': 0, 'weight':1},
    #                  {'loc': (MAP_SIZE[0] // 2, MAP_SIZE[1] - 1), 'color': (191, 90, 17), 'scale': 0,'weight':1},
    #                  {'loc': (MAP_SIZE[0] // 2, MAP_SIZE[0] // 2), 'color': (255, 255, 255), 'scale': 0,'weight':0}]
    # all_pts = [i['loc'] for i in initial_color]
    # for d in initial_color:
    #     _y, _x = d['loc']
    #     d['scale'] = np.max([dist((_y, _x), (0, 0)),
    #                         dist((_y, _x), (MAP_SIZE[0] - 1, 0)),
    #                         dist((_y, _x), (0, MAP_SIZE[1] - 1)),
    #                         dist((_y, _x), (MAP_SIZE[0] - 1, MAP_SIZE[1] - 1))])
    # img = np.zeros((MAP_SIZE[0], MAP_SIZE[1], 3),dtype=np.float32)
    # for i in initial_color:
    #     img += make_img(i['color'],i['loc'], i['scale'], MAP_SIZE) * i['weight']
    # np.clip(img, 0, 255, img)
    # img = np.round(img).astype(np.uint8)
    # display(img,encode='RGB')

    start_time = time()
    img = cv2.imread(r'Data\mask\0_texture.png')
    edge_img, edge_along_y, edge_along_x = edge_detection(img, th=5)
    new_img = color_gradient_v4(img, edge_along_y)
    display(new_img, "v4")
