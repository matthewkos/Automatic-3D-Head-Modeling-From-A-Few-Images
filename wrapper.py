import subprocess
import os
import argparse
import json
import subprocess

def blender_wrapper(blender_file, script_file_path, input_data,texture,hair,mask, output):
    # LOAD CONFIG FILE
    with open('.\config.ini', 'r') as json_file:
        config = json.load(json_file)
    config['INPUT_DATA'] = "r'{}'".format(input_data)
    config['TEXTURE_DATA'] = "r'{}'".format(texture)
    config['HAIR_DATA'] = "r'{}'".format(hair)
    config['MASK_DATA'] = "r'{}'".format(mask)
    config['OUT_DATA'] = "r'{}'".format(output)
    json.dump(config, open('.\config.ini', 'w'))
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("script_file_path", help="the blender script path")
    args = parser.parse_args()
    blender_wrapper(args.script_file_path)
