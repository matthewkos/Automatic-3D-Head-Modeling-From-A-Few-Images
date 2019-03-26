from time import time
import warnings
import os
import subprocess
from DataHelper import ConfigManager

def blender_wrapper(blender_file, script_file_path, input_data, texture, hair, mask, output):
    # LOAD CONFIG FILE
    configManager = ConfigManager('.\\config.ini')
    keyAndValue = {}
    keyAndValue['INPUT_DATA'] = input_data
    keyAndValue['TEXTURE_DATA'] = texture
    keyAndValue['HAIR_DATA'] = hair
    keyAndValue['MASK_DATA'] = mask
    keyAndValue['OUT_DATA'] = output
    configManager.addPairs(keyAndValue)
    # SAVE CONFIG FILE

    blender = r".\\Blender\\blender.exe"
    if os.path.exists(blender):
        cmd = "{} -b {} -P {}".format(blender, blender_file, script_file_path)
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


if __name__ == '__main__':
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
    del json_data

    """Setup"""
    warnings.filterwarnings("ignore")
    print("Importing packages: ")
    start_time = time()
    from PRNet.myPRNET import genPRMask

    print("\ttime={:.2f}s".format(time() - start_time))
    """END"""

    path = input("Path of image: ")
    # path = INPUT_DATA
    path = os.path.join(DIR_INPUT, path)
    assert os.path.exists(path)

    time_it_wrapper(None, "Generating Geometry")
    time_it_wrapper(None, "Generating Texture")
    mask_data = time_it_wrapper(genPRMask, "Generating Mask", (path, DIR_MASK))
    # MASK_DATA = mask_data['obj']
# time_it_wrapper(
#     blender_wrapper(".\\geometry.blend", ".\\blender_script\\geo.py", INPUT_DATA, TEXTURE_DATA, HAIR_DATA,
#                     MASK_DATA, OUT_DATA),
#     "Alignment and Export")
time_it_wrapper(blender_wrapper, "Alignment",
                args=(".\\geometry.blend", ".\\blender_script\\geo.py", INPUT_DATA, TEXTURE_DATA, HAIR_DATA,
                      MASK_DATA, OUT_DATA))
print("Total_time: {:.2f}".format(time() - global_start))
