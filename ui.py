import PySimpleGUI as sg
import os
import shutil
import pyglet
from datetime import datetime
from PIL import Image
from MeshViewer import MeshViewer
from ConfigManager import ConfigManager


class ImageViewer(object):

    def __init__(self, filename_input=None):
        self.layout = [[sg.Text('Imported Images')],
                       [sg.Image(filename=filename_input, key='2D_IMAGE_VIEWER')],
                       [sg.Button('Exit')]
                       ]
        self.window = sg.Window('ImageViewer')
        self.window.Layout(self.layout)
    def run(self):
        while True:
            ev2, _ = self.window.Read()
            if ev2 is None or ev2 == 'Exit':
                self.window.Close()
                break

if __name__ == '__main__':
    # Load JSON File Data
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

    sg.ChangeLookAndFeel('DarkBlue')

    layout = [
        [sg.Text('Input Image Path:'), sg.Text('', key='_IMG_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('2D Frontal Image ', size=(15, 1)),
         sg.InputText('C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\input\\0.jpg', key='_IMG_PATH_'), sg.FileBrowse()],
        [sg.Button('Show Image')],
        [sg.Text('')],
        [sg.Button('Generate')],
        [sg.Text('')],
        [sg.Text('Output 3D Head Obj Path:'), sg.Text('', key='_OBJ_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('3D Head WaveFront ', size=(15, 1)), sg.InputText('C:\\Users\\KTL\\Desktop\\FYP-code\\output', key='_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Button('Show 3D model')],
        [sg.Button('Exit')]
    ]

    window_main = sg.Window('Automatic Head Modelling').Layout(layout)
    imageViewer_active = False

    current_img_path = ''
    current_obj_path = ''
    while True:  # Event Loop
        event, values = window_main.Read()
        print(event, values)
        if event is None or event == 'Exit':
            break
        if event == 'Show Image':
            try:
                current_img_path = values['_IMG_PATH_'] if values['_IMG_PATH_'] != '' else current_img_path
                current_img_path = os.path.normcase(current_img_path)
                print('current_img_path = ', current_img_path)
                if not os.path.exists(current_img_path):
                    raise ValueError('Invalid Path')

                INPUT_DATA = os.path.split(current_img_path)[-1]

                if os.path.isfile(".\\"+DIR_INPUT + ".\\" + INPUT_DATA):
                    # Rename the new input file if the designated path has a file with same filename
                    dot_index = INPUT_DATA.find('.')
                    INPUT_DATA = INPUT_DATA[0:dot_index] + '_{0:%Y%m%d_%H%M%S}'.format(datetime.now()) + INPUT_DATA[dot_index:]

                if '.png' not in current_img_path:
                    # Convert to .png for view
                    im = Image.open(current_img_path)
                    if not os.path.exists(".\\" + DIR_INPUT):
                        os.mkdir(".\\"+DIR_INPUT)
                    display_img_path = ".\\"+DIR_INPUT + ".\\" + INPUT_DATA.replace('.jpg', '.png')
                    display_img_path = os.path.abspath(display_img_path)
                    print('display_img_path = ', display_img_path)
                    im.save(display_img_path)
                elif '.jpg' not in current_img_path:
                    # INPUT File can be in .jpg/.png file format
                    pass
                
               
                # Copy the image file to the designated path of config.ini   
                new_img_path = ".\\"+DIR_INPUT + ".\\" + INPUT_DATA     
                shutil.copy(current_img_path, new_img_path)
                current_img_path = os.path.abspath(new_img_path)
                
                # Change the input image path in config.ini
                configManager.addOne('INPUT_DATA', INPUT_DATA)
                print('current_img_path = ', current_img_path)

                window_main.FindElement('_IMG_PATH_').Update(current_img_path)
                # Show another window for the 2D frontal image
                window_main.FindElement('_IMG_PATH_DISPLAY_').Update(display_img_path)
                imageViewer = ImageViewer(filename_input=display_img_path)
                # imageViewer_active = True
                # window_main.Hide()
                imageViewer.run()
                # imageViewer_active = False
                # window_main.UnHide()
            except Exception as err:
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] +"\n" + errmsg[100:]
                sg.PopupError(errmsg)
        elif event == 'Generate':
            # print(event)
            pass
            img_path = values['_IMG_PATH_'] if values['_IMG_PATH_'] != '' else current_img_path
            # main.gen(img_path)
        elif event == 'Show 3D model':
            try:
                current_obj_path = values['_OBJ_PATH_'] if values['_OBJ_PATH_'] != '' else current_obj_path
                window_main.FindElement('_OBJ_PATH_DISPLAY_').Update(current_obj_path)
                if not os.path.exists(current_img_path):
                    window = MeshViewer(current_obj_path)
                    pyglet.clock.schedule(window.update)
                    pyglet.app.run()
                    window.show_window()
                else:
                    raise ValueError('Invalid Path')
            except Exception as err:
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] + "\n" + errmsg[100:]
                window_main.FindElement('_ERROR_MSG_').Update(errmsg)
    window_main.Close()
