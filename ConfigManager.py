import os
import json

class ConfigManager(object):

    def __init__(self, configFilePath):
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