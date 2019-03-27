from time import time
import warnings
import os
import subprocess
from DataHelper import ConfigManager
import cv2
import numpy as np
from math import sqrt

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


def image_expansion(img, mode=''):
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
            length = img.shape[1] - _x1
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
    _y1 = np.argwhere(edges_along_y[:, 0] > 0).max()
    for _x in range(img.shape[1]):
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
            length = img.shape[0] - _y1
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


def genText(img_path, output_path):
    img = cv2.imread(img_path)
    # 256 * 256 -> 1024*256
    new_img = np.zeros((256, 512, 3), dtype=np.uint8)
    new_img[:, (512-256)//2:(512+256)//2, :] = img
    img = image_expansion(new_img, 'i')
    cv2.imwrite(output_path, img)
    return


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
    time_it_wrapper(genPRMask, "Generating Mask", (os.path.join(DIR_INPUT, img_path), DIR_MASK))
    """Texture"""
    time_it_wrapper(genText, "Generating Texture", (
        os.path.join(DIR_MASK, "{}_texture.png".format(MASK_DATA[:-4])), os.path.join(DIR_TEXTURE, TEXTURE_DATA)))
    # display(os.path.join(DIR_TEXTURE, TEXTURE_DATA))
    """Alignment"""
    time_it_wrapper(blender_wrapper, "Alignment",
                    args=(".\\geometry.blend", ".\\blender_script\\geo.py", INPUT_DATA, TEXTURE_DATA, HAIR_DATA,
                          MASK_DATA, OUT_DATA, False, False))
    print("Output to: {}".format(os.path.join(os.getcwd(), DIR_OUT, OUT_DATA)))
    print("Total_time: {:.2f}".format(time() - global_start))
    return


if __name__ == '__main__':
    main()
