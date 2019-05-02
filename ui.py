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
import webbrowser


class ImageViewer(object):

    def __init__(self, filename_input=None):
        self.layout = [[sg.Text('File Path: ' + filename_input)],
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
    OBJ_HEAD_MODEL_HAIR = json_data.get("OBJ_HEAD_MODEL_HAIR", "Data\\hair\\head_model.obj")
    DIR_INPUT = json_data.get("DIR_INPUT", "Data\\input")
    DIR_TEXTURE = json_data.get("DIR_TEXTURE", "Data\\texture")
    DIR_HAIR = json_data.get("DIR_HAIR", "Data\\hair")
    DIR_MASK = json_data.get("DIR_MASK", "Data\\mask")
    DIR_OUT = json_data.get("DIR_OUT", "output")
    DIR_KPTS = json_data.get("DIR_KPTS", "Data\\geometry")
    INPUT_DATA = json_data.get("INPUT_DATA", "test.jpg")
    TEXTURE_DATA = json_data.get("DIR_KPTS", "test.jpg")
    HAIR_DATA = json_data.get("HAIR_DATA", "strands00260.data")
    MASK_DATA = json_data.get("MASK_DATA", "test.obj")
    OUT_DATA = json_data.get("OUT_DATA", "test.obj")
    HAIR = json_data.get('HAIR', True)
    HAIR_COLOR = json_data.get('HAIR_COLOR', [0, 0, 0])
    UI_DISPLAY_WIDTH = json_data.get("UI_DISPLAY_WIDTH", 128)
    UI_DISPLAY_HEIGHT = json_data.get("UI_DISPLAY_HEIGHT", 128)
    BLENDER_BACKGROUND = json_data.get("BLENDER_BACKGROUND", False)
    del json_data

    #################### Setup #######################

    warnings.filterwarnings("ignore")

    ##################################################

    sg.ChangeLookAndFeel('Black')
    DEFAULT_INPUT = os.path.abspath(os.path.join(DIR_INPUT, INPUT_DATA))
    DEFAULT_INPUT_DISPLAY = os.path.abspath(os.path.join(".temp", INPUT_DATA.replace('.jpg', '.png')))
    if not os.path.exists(DEFAULT_INPUT_DISPLAY):
        DEFAULT_INPUT_DISPLAY = os.path.abspath(r".\.temp\default_white.png")
    # TODO: Confirm the default INPUT and OUTPUT
    DEFAULT_OUTPUT = os.path.abspath(DIR_OUT)
    DEFAULT_HEAD_OUTPUT = None
    DEFAULT_HAIR_OUTPUT = None

    # Menu Frame Layout
    # Menu Definition
    menu_bar_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'Exit']],
                    # ['&Edit', ['Redo', 'Undo'], ],
                    ['&Help', ['About'], ]
    ]

    # Input Frame Layout
    input_frame_layout = [
        [sg.Text('Input Image Path:', size=(20, 1)), sg.Text(DEFAULT_INPUT, key='_IMG_PATH_DISPLAY_', size=(50, 1))],
        [sg.Text('2D Frontal Image ', size=(20, 1)), sg.InputText(DEFAULT_INPUT, size=(50, 1), key='_IMG_PATH_'),
         sg.FileBrowse()],
        [sg.Button('Preview')]
    ]

    hairstyle_preview_frame_layout = [
        [sg.Image(filename="Data\\ui_images\\strands00001.png", size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT),
                  key='_HAIR_PREVIEW_1_', visible=True)],
        [sg.Slider(range=(1, 343), key='_HAIRSTYLE_PREVIEW_SLIDER_', orientation='h', enable_events=True,
                   disable_number_display=False, size=(20, 15), font=("Helvetica", 10))],
        [sg.Text('No.:', size=(5, 1)), sg.InputText('1', key='_HAIR_NO_INPUT_', size=(15, 1))],
        [sg.Button('Select', key='_SELECT_HAIR_NO_')]
    ]

    # Image Frame Layout
    input_preview_frame_layout = [
        [sg.Image(filename=DEFAULT_INPUT_DISPLAY, size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT), key='_IMAGE_PREVIEW_',
                  visible=True)],
    ]

    # Generation Panel Frame Layout
    generation_panel_frame_layout = [
        [sg.Radio('Full Model', group_id="Generation_Setting", key="_full_model_radio_", default=True), ],
        [sg.Radio('Head Only', group_id="Generation_Setting", key="_head_only_radio_", size=(10, 1)), ],
        # [sg.Radio('Hair Only', group_id="Generation_Setting", key="_hair_only_radio_", size=(10, 1)), ],
        # [sg.Text('Hair Model (.obj) File Path:', size=(20, 1)),
        #  sg.InputText(DEFAULT_HAIR_OUTPUT, size=(50, 1), key='_HAIR_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Text('Output File Directory Path:', size=(20, 1)),
         sg.InputText(DEFAULT_OUTPUT, size=(50, 1), key='_HEAD_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Checkbox('Run Blender in  background', key="_blender_background_", default=False)],
        [sg.Button('Generate')]
    ]

    model_preview_frame_layout = [
        [sg.Text('Output 3D Head .obj File Path:'), sg.Text('', key='_OBJ_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('3D Head WaveFront ', size=(20, 1)), sg.InputText(DEFAULT_OUTPUT, size=(50, 1), key='_OBJ_PATH_'),
         sg.FileBrowse()],
        [sg.Button('Show 3D model')]
    ]

    hair_color_frame_layout = [
        [sg.Text('Hair Color:'), sg.InputText("#000000", key="_hair_color_value_")],
        [sg.ColorChooserButton("Choose", target="_hair_color_value_", key="_hair_color_chooser_")]
    ]
    # Main Layout 
    layout = [
        [sg.Menu(menu_bar_def)],
        [sg.Frame('Input', input_frame_layout, title_color='white', key="_INPUT_FRAME_", visible=True),
         sg.Frame('Input Preview', input_preview_frame_layout, title_color='white', key="_IMG_PREVIEW_FRAME_",
                  visible=True), ],
        [sg.Frame('Generation Control', generation_panel_frame_layout, title_color='white',
                  key="_GENERATION_CONTROL_FRAME_", visible=True),
         sg.Frame('Hairstyle Preview', hairstyle_preview_frame_layout, title_color='white'), ],
        [sg.Frame('3D Model Preview', model_preview_frame_layout, title_color='white', key="_MODEL_PREVIEW_FRAME_",
                  visible=True),
         sg.Frame('Hair Color', hair_color_frame_layout, title_color='white', key="_HAIR_COLOR_FRAME_",
                  visible=True)],

    ]
    # default window size = (698, 426)
    window_main = sg.Window('Automatic Head Modelling', size=(850, 550), icon='Data\\ui_images\\icon.ico',
                            auto_size_text=True, auto_size_buttons=True, resizable=True, grab_anywhere=False, ).Layout(
        layout)

    imageViewer_active = False

    current_img_path = ''
    current_obj_path = ''
    github_url = "https://github.com/fyp-sb3-mxj3/d50826df94911c26dc5e2f3db6f2fea2/tree/ludwig"

    ########## load PRNET ##########
    print("Importing packages: ")
    start_time = time()
    try:
        from PRNet.myPRNET import genPRMask
    except Exception as err:
        print(err)
        print("Cannot import PRNet. Please install all required packages in requirement.txt.")
        print("pip install -r requirement.txt")
    print("\ttime={:.2f}s".format(time() - start_time))
    ###############################

    while True:  # Event Loop
        event, values = window_main.Read()
        print(event, values)
        if event is None or event == 'Exit':
            break

        elif event == 'Open' or event == 'Save' or event == 'Properties':
            sg.Popup('Currently Unavailable')
        
        elif event == 'About':
            about_return_val = sg.PopupOKCancel('''
            Automatic 3D Head Modeling From A Few Images
            CHEUNG Yik Kin, KOO Tin Lok , OR Ka Po @ Hong Kong University of Science and Technology
            Final Year Project - ELEC4900 SB3-18 | COMP4901 MXJ3
            Please click OK to browse the Github repository for more information.
            ''')
            if about_return_val =='OK':
                webbrowser.open_new_tab(github_url)

        elif event == 'Preview':
            """
            Preview
            """
            try:
                current_img_path = values['_IMG_PATH_'] if values['_IMG_PATH_'] != '' else current_img_path
                print('current_img_path = ', current_img_path)
                if not os.path.isfile(current_img_path):
                    raise ValueError('Invalid Path')

                input_data = os.path.split(current_img_path)[-1]
                current_img_path = os.path.abspath(os.path.join(DIR_INPUT, input_data))
                # convert to png file
                if '.png' not in current_img_path:
                    # Try to find a .png file with the same name in the image directory
                    display_img_path = os.path.join(".temp", input_data.replace('.jpg', '.png'))
                    display_img_path = os.path.abspath(display_img_path)
                    if not os.path.isfile(display_img_path):
                        # convert to .png for view
                        im = Image.open(current_img_path)
                        im = im.resize((UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT), Image.ANTIALIAS)
                        im.save(display_img_path)
                else:
                    # Use the input .png file
                    display_img_path = current_img_path

                window_main.FindElement('_IMG_PATH_').Update(current_img_path)
                window_main.FindElement('_IMG_PATH_DISPLAY_').Update(current_img_path)
                window_main.FindElement('_HEAD_OBJ_PATH_').Update(values['_HEAD_OBJ_PATH_'])
                window_main.FindElement('_OBJ_PATH_').Update(values['_OBJ_PATH_'])
                window_main.FindElement('_HAIR_NO_INPUT_').Update(str(int(values['_HAIRSTYLE_PREVIEW_SLIDER_'])))
                window_main.FindElement('_IMAGE_PREVIEW_').Update(filename=display_img_path,
                                                                  size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT),
                                                                  visible=True)
                window_main.FindElement('_hair_color_value_').Update(values['_hair_color_value_'])

            except Exception as err:
                # Display the error by popup window
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] + "\n" + errmsg[100:]
                sg.PopupError(errmsg)

        elif event == '_HAIRSTYLE_PREVIEW_SLIDER_':
            """
            Preview hairstyle
            """
            # Slide to change hairstyle preview images
            slider_value = values['_HAIRSTYLE_PREVIEW_SLIDER_']
            hair_file_name = "strands{}.data".format(str(int(slider_value)).zfill(5))
            if os.path.exists(os.path.join(DIR_HAIR, hair_file_name)):
                window_main.FindElement('_HAIR_PREVIEW_1_').Update(
                    os.path.join("Data", "ui_images", hair_file_name[:-5] + ".png"),
                    size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT))
                window_main.FindElement('_IMG_PATH_').Update(values['_IMG_PATH_'])
                window_main.FindElement('_HEAD_OBJ_PATH_').Update(values['_HEAD_OBJ_PATH_'])
                window_main.FindElement('_OBJ_PATH_').Update(values['_OBJ_PATH_'])
                window_main.FindElement('_HAIR_NO_INPUT_').Update(str(int(values['_HAIRSTYLE_PREVIEW_SLIDER_'])))
                window_main.FindElement('_hair_color_value_').Update(values['_hair_color_value_'])

        elif event == '_SELECT_HAIR_NO_':
            """
            Select Hairstyle by inputing the number
            """
            slider_value = values['_HAIR_NO_INPUT_']
            hair_file_name = "strands{}.data".format(str(int(slider_value)).zfill(5))
            if os.path.exists(os.path.join(DIR_HAIR, hair_file_name)):
                window_main.FindElement('_HAIR_PREVIEW_1_').Update(
                    os.path.join("Data", "ui_images", hair_file_name[:-5] + ".png"),
                    size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT))
                # Update Slider's value
                window_main.FindElement('_IMG_PATH_').Update(values['_IMG_PATH_'])
                window_main.FindElement('_HEAD_OBJ_PATH_').Update(values['_HEAD_OBJ_PATH_'])
                window_main.FindElement('_OBJ_PATH_').Update(values['_OBJ_PATH_'])
                window_main.FindElement('_HAIR_NO_INPUT_').Update(str(int(slider_value)))
                window_main.FindElement('_HAIRSTYLE_PREVIEW_SLIDER_').Update((int(slider_value)))
                window_main.FindElement('_hair_color_value_').Update(values['_hair_color_value_'])

        elif event == 'Generate':
            """
            Generate
            """
            try:
                tf.reset_default_graph()
                # Update the input image path in config.ini

                current_img_path = values['_IMG_PATH_'] if values['_IMG_PATH_'] != '' else current_img_path
                # OUT_DIR = values['']
                if not os.path.isfile(current_img_path):
                    raise ValueError('Invalid Path')
                slider_value = values['_HAIRSTYLE_PREVIEW_SLIDER_']
                HAIR_DATA = "strands{}.data".format(str(int(slider_value)).zfill(5))
                if os.path.exists(os.path.join(DIR_HAIR, HAIR_DATA)):
                    configManager.addOne('HAIR_DATA', HAIR_DATA)
                    window_main.FindElement('_HAIR_PREVIEW_1_').Update(
                        os.path.join("Data", "ui_images", HAIR_DATA[:-5] + ".png"),
                        size=(UI_DISPLAY_WIDTH, UI_DISPLAY_HEIGHT))
                INPUT_DATA = input_data = os.path.split(current_img_path)[-1]
                TEXTURE_DATA = input_data[:-4] + '.jpg'
                MASK_DATA = input_data[:-4] + '.obj'
                OUT_DATA = input_data[:-4] + '.obj'
                DIR_OUT = values['_HEAD_OBJ_PATH_']
                configManager.addOne('DIR_OUT', DIR_OUT.split("\\")[-1])
                configManager.addOne('INPUT_DATA', input_data)
                configManager.addOne('TEXTURE_DATA', TEXTURE_DATA)
                configManager.addOne('MASK_DATA', MASK_DATA)
                configManager.addOne('OUT_DATA', OUT_DATA)
                if current_img_path == "":
                    # use default image if user does not input a path before
                    print("Empty image path input.")
                    print("Use default image path:", DEFAULT_INPUT)
                    current_img_path = DEFAULT_INPUT

                # Change to relative path
                relative_current_img_path = os.path.relpath(current_img_path, os.getcwd())
                assert os.path.exists(relative_current_img_path), "Invalid path: " + relative_current_img_path

                # Update the BLENDER_BACKGROUND setting in config.ini
                configManager.addOne('BLENDER_BACKGROUND', values["_blender_background_"])
                BLENDER_BACKGROUND = values["_blender_background_"]
                HAIR = None
                if values["_full_model_radio_"]:
                    HAIR = True
                elif values["_head_only_radio_"]:
                    HAIR = False
                configManager.addOne('HAIR', HAIR)
                HAIR_DATA = "strands{}.data".format(str(int(values['_HAIRSTYLE_PREVIEW_SLIDER_'])).zfill(5))
                configManager.addOne('HAIR_DATA', HAIR_DATA)
                hair_color_value = values['_hair_color_value_'][1:]
                HAIR_COLOR = [int(hair_color_value[a:a + 2], 16) for a in [0, 2, 4]]
                configManager.addOne('HAIR_COLOR', HAIR_COLOR)

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
                    os.path.join(DIR_MASK, "{}_texture.png".format(MASK_DATA[:-4])),  # input full
                    os.path.join(DIR_MASK, "{}_texture_2.png".format(MASK_DATA[:-4])),  # input half
                    os.path.join(DIR_TEXTURE, TEXTURE_DATA),  # output texture for head
                    os.path.join(DIR_MASK, "{}_texture.png".format(MASK_DATA[:-4])),  # output texture for mask
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
                    HAIR_COLOR,
                    BLENDER_BACKGROUND))
                print("Output to: {}".format(os.path.join(os.getcwd(), DIR_OUT, OUT_DATA)))
                print("Total_time: {:.2f}".format(time() - global_start))

                # After Generation:
                current_obj_path = os.path.join(DIR_OUT, configManager.getOne("OUT_DATA"))
                current_obj_path = os.path.abspath(current_obj_path)
                window_main.FindElement('_OBJ_PATH_DISPLAY_').Update(current_obj_path)
                window_main.FindElement('_IMG_PATH_').Update(values['_IMG_PATH_'])
                window_main.FindElement('_HEAD_OBJ_PATH_').Update(values['_HEAD_OBJ_PATH_'])
                window_main.FindElement('_OBJ_PATH_').Update(values['_OBJ_PATH_'])
                window_main.FindElement('_HAIR_NO_INPUT_').Update(str(int(slider_value)))
                window_main.FindElement('_hair_color_value_').Update(values['_hair_color_value_'])
            except Exception as e:
                print("Error:", e)
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
