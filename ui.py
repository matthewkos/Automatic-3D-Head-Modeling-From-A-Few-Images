import PySimpleGUI as sg
import os
import shutil
import pyglet
from datetime import datetime
from PIL import Image
from MeshViewer import MeshViewer
from DataHelper import ConfigManager

class ImageViewer(object):

    def __init__(self, filename_input=None):
        self.layout = [[sg.Text('File Path: '+filename_input)],
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
    UI_DISPLAY_WIDTH = json_data["UI_DISPLAY_WIDTH"]
    UI_DISPLAY_HEIGHT = json_data["UI_DISPLAY_HEIGHT"]
    del json_data    

    sg.ChangeLookAndFeel('DarkBlue')
    DEFAULT_INPUT = os.path.abspath(".\\" + DIR_INPUT + "\\" + INPUT_DATA)
    DEFAULT_INPUT_DISPLAY = os.path.abspath(".\\" + DIR_INPUT + "\\" + INPUT_DATA.replace('.jpg', '.png'))
    # TODO: Confirm the default INPUT and OUTPUT
    DEFAULT_OUTPUT = os.path.abspath(".\\" + DIR_OUT + "\\" + OUT_DATA)
    
    # Input Frame Layout
    input_frame_layout = [
         [sg.Text('Input Image Path:', size=(15, 1)), sg.Text(DEFAULT_INPUT, key='_IMG_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('2D Frontal Image ', size=(15, 1)), sg.InputText(DEFAULT_INPUT, key='_IMG_PATH_'), sg.FileBrowse()],
        [sg.Button('Preview')]
    ]

    # Image Frame Layout
    input_preview_frame_layout = [
                  [sg.Image(filename=DEFAULT_INPUT_DISPLAY ,size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT), key='_IMAGE_PREVIEW_', visible=False)],      
               ]  

    # Generation Panel Frame Layout
    generation_panel_frame_layout =[
        [sg.Radio('Full Model', group_id="Generation_Setting", key="_full_model_radio_", default=True), sg.Radio('Head Only', group_id="Generation_Setting", key="_head_only_radio_"), sg.Radio('Hair Only', group_id="Generation_Setting", key="_hair_only_radio_")],
        [sg.Button('Generate')]
    ]  

    model_preview_frame_layout = [
        [sg.Text('Output 3D Head .obj File Path:'), sg.Text('', key='_OBJ_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('3D Head WaveFront ', size=(15, 1)), sg.InputText(DEFAULT_OUTPUT, key='_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Button('Show 3D model')]
    ]

    # Main Layout 
    layout = [
        [sg.Frame('Input', input_frame_layout, size=(15, 2), title_color='blue', key="_INPUT_FRAME_", visible=True)],
        [sg.Frame('Input Preview', input_preview_frame_layout, size=(15, 2), title_color='blue', key="_IMG_PREVIEW_FRAME_", visible=False)],
        [sg.Frame('Generation Control', generation_panel_frame_layout, size=(15, 2), title_color='blue', key="_GENERATION_CONTROL_FRAME_", visible=True)],
        [sg.Frame('3D Model Preview', model_preview_frame_layout, size=(15, 2), title_color='blue', key="_MODEL_PREVIEW_FRAME_", visible=True)],
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
        if event == 'Preview':
            try:
                current_img_path = values['_IMG_PATH_'] if values['_IMG_PATH_'] != '' else current_img_path
                print('current_img_path = ', current_img_path)
                if not os.path.isfile(current_img_path):
                    raise ValueError('Invalid Path')

                input_data = os.path.split(current_img_path)[-1]

                if os.path.isfile(".\\"+DIR_INPUT + ".\\" + input_data):
                    if not os.path.samefile(".\\"+DIR_INPUT + ".\\" + input_data, current_img_path):
                        # Rename the new input file if the designated path has a file with same filename
                        dot_index = input_data.find('.')
                        input_data = input_data[0:dot_index] + '_{0:%Y%m%d_%H%M%S}'.format(datetime.now()) + input_data[dot_index:]
                        # Copy the image file to the designated path of config.ini   
                        new_img_path = ".\\"+DIR_INPUT + ".\\" + input_data     
                        shutil.copy(current_img_path, new_img_path)
                        current_img_path = os.path.abspath(new_img_path)
                    else:
                        current_img_path = os.path.abspath(".\\"+DIR_INPUT + ".\\" + input_data)

                if '.png' not in current_img_path:
                    # Try to find a .png file with the same name in the image directory
                    display_img_path = ".\\"+DIR_INPUT + ".\\" + input_data.replace('.jpg', '.png')
                    display_img_path = os.path.abspath(display_img_path)
                    if not os.path.isfile(display_img_path):
                        # convert to .png for view
                        im = Image.open(current_img_path)
                        im = im.resize((UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT),Image.ANTIALIAS)
                        im.save(display_img_path)
                else:
                    # Use the input .png file
                    display_img_path = current_img_path
                    pass
                print('display_img_path = ', display_img_path)
            
                # Change the input image path in config.ini
                configManager.addOne('INPUT_DATA', input_data)
                print('current_img_path = ', current_img_path)

                window_main.FindElement('_IMG_PATH_').Update(current_img_path)
                # Show another window for the 2D frontal image
                window_main.FindElement('_IMG_PATH_DISPLAY_').Update(display_img_path)
                # imageViewer = ImageViewer(filename_input=display_img_path)
                # imageViewer.run()

                window_main.FindElement('_IMAGE_PREVIEW_').Update(filename=display_img_path, size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT), visible=True)
                window_main.FindElement('_IMG_PREVIEW_FRAME_').Update(visible=True)
            except Exception as err:
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] +"\n" + errmsg[100:]
                sg.PopupError(errmsg)
        elif event == 'Generate':
            if current_img_path == "":
                # use default image if user does not input a path before
                pass
            if values["_full_model_radio_"]:
                # Generate complete model
                pass
            elif values["_head_only_radio_"]:
                # Generate head model only
                pass
            elif values["_hair_only_radio_"]:
                # Generate hair model only
                pass
            # main.gen(DEFAULT_INPUT)

            # After Generation:
            current_obj_path = ".\\" + DIR_OUT + "\\" + configManager.getOne("OUT_DATA")
            current_obj_path = os.path.abspath(current_obj_path)
            window_main.FindElement('_OBJ_PATH_DISPLAY_').Update(current_obj_path)
               
        elif event == 'Show 3D model':
            try:
                current_obj_path = values['_OBJ_PATH_'] if values['_OBJ_PATH_'] != '' else current_obj_path
                window_main.FindElement('_OBJ_PATH_DISPLAY_').Update(current_obj_path)
                if os.path.isfile(current_obj_path):
                    window = MeshViewer(current_obj_path)
                    pyglet.clock.schedule(window.update)
                    pyglet.app.run()
                    window.show_window()
                else:
                    raise ValueError('Invalid Path')
            except Exception as err:
                event_loop = pyglet.app.EventLoop()
                event_loop.exit()
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] + "\n" + errmsg[100:]
                sg.PopupError(errmsg)
    window_main.Close()
