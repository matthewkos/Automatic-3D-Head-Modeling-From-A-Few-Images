import os

dir_ui = "Data/ui_images"
dir_hair = "Data/hair"

# os.chdir(os.path.abspath(dir_hair))
# print(os.getcwd())
# data = [int(x[7:-5]) for x in os.listdir() if x.endswith('.data') ]
# file_name = [x for x in os.listdir() if x.endswith('.data') ]
# print(file_name)
# for i, name in enumerate(file_name):
#     new_name = "strands{}.data".format(str(i+1).zfill(5))
#     print(i+1,name, new_name)
#     os.rename(name,new_name)

os.chdir(os.path.abspath(dir_ui))
print(os.getcwd())
# data = [int(x[7:-5]) for x in os.listdir() if x.startswith('strands')]
file_name = [x for x in os.listdir() if x.startswith('strands') ]
print(file_name)
for i, name in enumerate(file_name):
    new_name = "strands{}.png".format(str(i+1).zfill(5))
    print(i+1,name, new_name)
    os.rename(name,new_name)