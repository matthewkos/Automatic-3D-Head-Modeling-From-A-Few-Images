import os
import json

class ConfigManager(object):

    def __init__(self, configFilePath='.\\config.ini'):
        self.configFilePath = configFilePath
        if not os.path.exists(self.configFilePath):
            # Default config.ini path
            self.configFilePath = '.\\config.ini'

    def addOne(self, key, value):
        with open(self.configFilePath, 'r') as json_file:
            config = json.load(json_file)
            config[key] = value
        with open(self.configFilePath, "w") as json_file:
            json.dump(config, json_file, indent=4)

    def getOne(self, key):
        with open(self.configFilePath, 'r') as json_file:
            config = json.load(json_file)
            value = config[key]
        return value

    def addPairs(self, keyAndValue):
        with open(self.configFilePath, 'r') as json_file:
            config = json.load(json_file)
            for key, value in keyAndValue.items():
                config[key] = value
        with open(self.configFilePath, "w") as json_file:
            json.dump(config, json_file, indent=4)

    def getAll(self):
        with open(self.configFilePath, 'r') as json_file:
            config = json.load(json_file)
        return config

    def isEqualValue(self, key, value):
        with open(self.configFilePath, 'r') as json_file:
            config = json.load(json_file)
            if config[key] == value:
                return True
        return False

def convertFilePathInMTL(filename):
    # Convert the file path of texture image in .mtl to relative path
    with open(filename, "r") as mtl:
        lines = mtl.readlines()
    for i in range(len(lines)):
        if "map_Kd" in lines[i]:
            # print(lines[i].split())
            line = lines[i].split()
            line[1] = line[1].split("\\")[-1] + "\n"
            lines[i] = " ".join(line) 
            # print(lines[i])
            break
    with open(filename, "w") as mtl:
        mtl.writelines(lines)

def convertFilePathInOBJ(filename):
    # Convert the file path of .mtl in .obj to relative path
    with open(filename) as obj: 
        lines = obj.readlines()
    for i in range(len(lines)):
        if "mtllib" in lines[i]:
            print(lines[i].split())
            line = lines[i].split()
            line[1] = line[1].split("\\")[-1] + "\n"
            lines[i] = " ".join(line) 
            print(lines[i])
            break
    with open(filename, "w") as obj:
        obj.writelines(lines)

    

