import bpy
import json
print("Success" if bpy==bpy else "Fail")

with open('./config.ini','r') as json_file:
	json_data = json.load(json_file)
	for (k,v) in json_data.items():
		exec("{} = {}".format(k,v))
