from time import time
import warnings
import os
from wrapper import blender_wrapper
import json

def time_it_wrapper(callback, name="", args=(), kwargs={}):
    print(name, ": ", end="")
    start_time = time()
    temp = None
    if callback:
        temp = callback(*args, **kwargs)
    print("time={:.2f}s".format(time() - start_time))
    return temp

if __name__ == '__main__':
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

    print("time={:.2f}".format(time() - start_time))
    """END"""

    # path = input("Path of image: ")
    path = INPUT_DATA
    path = os.path.join(DIR_INPUT, path)
    assert os.path.exists(path)

    time_it_wrapper(None, "Generating Geometry")
    time_it_wrapper(genPRMask, "Generating Mask", (path, DIR_MASK))
