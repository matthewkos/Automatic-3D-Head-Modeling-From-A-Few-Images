import PySimpleGUI as sg
import concurrent.futures
import os
import shutil
import pyglet
import warnings
from time import time
from datetime import datetime
from PIL import Image
from MeshViewer import MeshViewer
from DataHelper import ConfigManager
from main import *
import tensorflow as tf

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

    #################### Setup #######################
    
    warnings.filterwarnings("ignore")
    print("Importing packages: ")
    start_time = time()
    try:
        from PRNet.myPRNET import genPRMask
    except Exception as err:
        print(err)
        print("Cannot import PRNet. Please install all required packages in requirement.txt.")
        print("pip install -r requirement.txt")
    print("\ttime={:.2f}s".format(time() - start_time))

    # create tensorflow sess
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    ##################################################

    sg.ChangeLookAndFeel('Black')
    DEFAULT_INPUT = os.path.abspath(".\\" + DIR_INPUT + "\\" + INPUT_DATA)
    DEFAULT_INPUT_DISPLAY = os.path.abspath(".\\" + DIR_INPUT + "\\" + INPUT_DATA.replace('.jpg', '.png'))
    # TODO: Confirm the default INPUT and OUTPUT
    DEFAULT_OUTPUT = os.path.abspath(".\\" + DIR_OUT + "\\" + OUT_DATA)
    DEFAULT_HEAD_OUTPUT = None
    DEFAULT_HAIR_OUTPUT = None
    
    # Menu Frame Layout
    # Menu Definition
    menu_bar_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit'  ]],      
                ['&Edit', ['Redo', 'Undo'],],      
                ['&Help', '&About...'],]    

    # Input Frame Layout
    input_frame_layout = [
        [sg.Text('Input Image Path:', size=(20, 1)), sg.Text(DEFAULT_INPUT, key='_IMG_PATH_DISPLAY_', size=(50, 1))],
        [sg.Text('2D Frontal Image ', size=(20, 1)), sg.InputText(DEFAULT_INPUT, size=(50, 1), key='_IMG_PATH_'), sg.FileBrowse()],
        [sg.Button('Preview'), sg.Button('Close', key='_close_preview_image_')]
    ]

    hairstyle_preview_frame_layout = [
        [sg.Image(filename="Data\\ui_images\\strands00001.png" ,size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT), key='_HAIR_PREVIEW_1_', visible=True),],
        [sg.Slider((1,50),  key='_HAIRSTYLE_PREVIEW_SLIDER_', orientation='h', enable_events=True, disable_number_display=False, size=(5,10), font=("Helvetica", 10))]
    ]

    # Image Frame Layout
    input_preview_frame_layout = [
        [sg.Image(filename=DEFAULT_INPUT_DISPLAY ,size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT), key='_IMAGE_PREVIEW_', visible=True)],
    ]  


  
    # Generation Panel Frame Layout
    generation_panel_frame_layout =[
        [sg.Radio('Full Model', group_id="Generation_Setting", key="_full_model_radio_", default=True), ],
        [sg.Radio('Head Only', group_id="Generation_Setting", key="_head_only_radio_", size=(10, 1)), ],
        [sg.Radio('Hair Only', group_id="Generation_Setting", key="_hair_only_radio_", size=(10, 1)), ],
        [sg.Text('Hair Model (.obj) File Path:', size=(20, 1)), sg.InputText(DEFAULT_HAIR_OUTPUT, size=(50, 1), key='_HAIR_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Text('Head Model (.obj) File Path:', size=(20, 1)), sg.InputText(DEFAULT_HEAD_OUTPUT, size=(50, 1), key='_HEAD_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Checkbox('Blender Background', key = "_blender_background_", default=True)],
        [sg.Button('Generate')]
    ]  

    model_preview_frame_layout = [
        [sg.Text('Output 3D Head .obj File Path:'), sg.Text('', key='_OBJ_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('3D Head WaveFront ', size=(20, 1)), sg.InputText(DEFAULT_OUTPUT, size=(50, 1), key='_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Button('Show 3D model')]
    ]

    # Main Layout 
    layout = [
        [sg.Menu(menu_bar_def)],
        [sg.Frame('Input', input_frame_layout,title_color='white', key="_INPUT_FRAME_", visible=True),sg.Frame('Input Preview', input_preview_frame_layout,title_color='white', key="_IMG_PREVIEW_FRAME_", visible=True),],
        [sg.Frame('Generation Control', generation_panel_frame_layout, title_color='white', key="_GENERATION_CONTROL_FRAME_", visible=True), sg.Frame('Hairstyle Preview', hairstyle_preview_frame_layout, title_color='white'),],
        [sg.Frame('3D Model Preview', model_preview_frame_layout, title_color='white', key="_MODEL_PREVIEW_FRAME_", visible=True)],

    ]
    # default window size = (698, 426)
    window_main = sg.Window('Automatic Head Modelling',size=(850, 500), icon='Data\\ui_images\\icon.ico',auto_size_text=True, auto_size_buttons=True, resizable=True, grab_anywhere=False,).Layout(layout)
    
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
            
                # Update the input image path in config.ini
                configManager.addOne('INPUT_DATA', input_data)
                print('current_img_path = ', current_img_path)

                window_main.FindElement('_IMG_PATH_').Update(current_img_path)
                # Show another window for the 2D frontal image
                window_main.FindElement('_IMG_PATH_DISPLAY_').Update(display_img_path)
                window_main.FindElement('_IMAGE_PREVIEW_').Update(filename=display_img_path, size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT), visible=True)
                window_main.FindElement('_IMG_PREVIEW_FRAME_').Update(visible=True)
            except Exception as err:
                # Display the error by popup window
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] +"\n" + errmsg[100:]
                sg.PopupError(errmsg)
        elif event == '_close_preview_image_':
            window_main.FindElement('_IMG_PREVIEW_FRAME_').Update(visible=False)

        elif event == '_HAIRSTYLE_PREVIEW_SLIDER_':
            # Slide to change hairstyle preview images
            slider_value = values['_HAIRSTYLE_PREVIEW_SLIDER_']
            window_main.FindElement('_HAIR_PREVIEW_1_').Update('Data\\ui_images\\strands00002.png',size=(UI_DISPLAY_WIDTH,UI_DISPLAY_HEIGHT))
        elif event == 'Generate':
            if current_img_path == "":
                # use default image if user does not input a path before
                print("Empty image path input.")
                print("Use default image path:", DEFAULT_INPUT)
                current_img_path = DEFAULT_INPUT

            # Change to relative path
            relative_current_img_path = os.path.relpath(current_img_path, os.getcwd())
            assert os.path.exists(relative_current_img_path), "Invalid path: "+relative_current_img_path
            print("Relative image path:", relative_current_img_path)

            # Update the BLENDER_BACKGROUND setting in config.ini
            configManager.addOne('BLENDER_BACKGROUND', values["_blender_background_"])
            BLENDER_BACKGROUND = values["_blender_background_"]

            if values["_full_model_radio_"]:
                HAIR = True
            elif values["_head_only_radio_"]:
                HAIR = False
            elif values["_hair_only_radio_"]:
                # Generate hair model only
                pass
            configManager.addOne('HAIR', HAIR)

            global_start = time()
            """Geometry"""
            time_it_wrapper(None, "Generating Geometry")
            """Mask"""
            time_it_wrapper(genPRMask, "Generating Mask", args=(
                os.path.join(DIR_INPUT, INPUT_DATA),
                DIR_MASK),
                            kwargs={'isMask': False})
            """Texture"""
            time_it_wrapper(genText, "Generating Texture", args=(
                os.path.join(DIR_MASK, "{}_texture_2.png".format(MASK_DATA[:-4])),
                os.path.join(DIR_TEXTURE, TEXTURE_DATA),
                os.path.join(DIR_MASK, "{}_texture.png".format(MASK_DATA[:-4])),
                (512, 512, 3)
            ))
            """Alignment"""
            time_it_wrapper(blender_wrapper, "Alignment", args=(
                ".\\new_geometry.blend",
                ".\\blender_script\\geo.py",
                INPUT_DATA,
                TEXTURE_DATA,
                HAIR_DATA,
                MASK_DATA,
                OUT_DATA,
                HAIR,
                BLENDER_BACKGROUND))
            print("Output to: {}".format(os.path.join(os.getcwd(), DIR_OUT, OUT_DATA)))
            print("Total_time: {:.2f}".format(time() - global_start))

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
