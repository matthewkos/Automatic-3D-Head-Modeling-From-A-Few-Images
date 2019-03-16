from time import time
import warnings
import os
from wrapper import blender_wrapper
import json


def clear_all_output():
    pass


def time_it_wrapper(callback, name="", args=(), kwargs={}):
    print(name, ": ", end="")
    start_time = time()
    temp = None
    if callback:
        temp = callback(*args, **kwargs)
    print("time={:.2f}s".format(time() - start_time))
    return temp


if __name__ == '__main__':
    global_start = time()
    """Import constants from config file"""
    with open('config.ini', 'r') as json_file:
        json_data = json.load(json_file)
        for (k, v) in json_data.items():
            exec("{} = {}".format(k, v))
    """Setup"""
    warnings.filterwarnings("ignore")
    print("Importing packages: ", end="")
    start_time = time()
    from PRNet.myPRNET import genPRMask
    print("time={:.2f}s".format(time() - start_time))
    """END"""

    path = input("Path of image: ")
    # path = INPUT_DATA
    path = os.path.join(DIR_INPUT, path)
    assert os.path.exists(path)

    time_it_wrapper(None, "Generating Geometry")
    time_it_wrapper(None, "Generating Texture")
    time_it_wrapper(genPRMask, "Generating Mask", (path, DIR_MASK))
    time_it_wrapper(
        blender_wrapper(".\\geometry.blend", ".\\blender_script\\geo.py", INPUT_DATA, TEXTURE_DATA, HAIR_DATA,
                        MASK_DATA, OUT_DATA),
        "Alignment and Export")
    print("Global_time: {:.2f}".format(time() - global_start))
