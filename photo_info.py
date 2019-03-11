from mtcnn.mtcnn import MTCNN
import cv2
import sys
import numpy as np
import random
import os
import warnings
from time import time


def get_key_points(img, detector=MTCNN()):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    return detector.detect_faces(img)


def display(img, name="Img", time=0):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def addpoint(img, pt, radius=2, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    cv2.circle(img, pt, radius, color, -1)
    return img


def addbox(img, coord, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=2):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    cv2.rectangle(img, *coord, color, thickness)
    return img


def xywh2xyxy(x, y, w, h):
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    return (x1, y1), (x2, y2)


def crop(img, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    img = img[y1:y2, x1:x2]
    return img


def time_it_wrapper(callback, name="", args=(), kwargs={}):
    print(name, ": ", end="")
    start_time = time()
    temp = None
    if callback:
        temp = callback(*args, **kwargs)
    print("time={:.2f}s".format(time() - start_time))
    return temp


def bg_remove(img, out=None):
    # == Parameters
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white

    mask = np.zeros(edges.shape)
    # cv2.fillConvexPoly(mask, max_contour[0], (255))

    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))

    # -- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    if out:
        cv2.imwrite(out, masked)
    display(img,"masked")
    return masked


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # path = input("Path of image: ")
    path = "0.jpg"
    assert os.path.exists(path)

    img = cv2.imread(path)
    points = time_it_wrapper(get_key_points, "MTCNN", (img,))[0]
    print(points)

    points_box = xywh2xyxy(*points['box'])
    img = addbox(img, points_box)
    for pt in points['keypoints'].values():
        img = addpoint(img, pt)
    display(img)
    bg_remove(cv2.imread(path))
    # img = crop(img, *points_box)
    # display(img)

