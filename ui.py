import PySimpleGUI as sg
import os
import pyglet
from PIL import Image
from MeshViewer import MeshViewer


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
    sg.ChangeLookAndFeel('DarkBlue')

    layout = [
        [sg.Text('Input Image Path:'), sg.Text('', key='_IMG_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('2D Frontal Image ', size=(15, 1)),
         sg.InputText('C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\input\\0.jpg', key='_IMG_PATH_'), sg.FileBrowse()],
        [sg.Button('Show Image')],
        [sg.Text('Output 3D Head Obj Path:'), sg.Text('', key='_OBJ_PATH_DISPLAY_', size=(45, 1))],
        [sg.Text('3D Head WaveFront ', size=(15, 1)), sg.InputText('C:\\Users\\KTL\\Desktop\\FYP-code\\output', key='_OBJ_PATH_'), sg.FileBrowse()],
        [sg.Button('Generate')],
        [sg.Text('', key='_ERROR_MSG_', size=(75,2),text_color='#FF0000')],
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
                window_main.FindElement('_IMG_PATH_').Update(current_img_path)
                if not os.path.exists(current_img_path):
                    raise ValueError('Invalid Path')
                if '.png' not in current_img_path:
                    # Convert .jpg to .png for view
                    im = Image.open(current_img_path)
                    if not os.path.exists(".\\.temp"):
                        os.mkdir(".\\.temp")
                    current_img_path = ".\\.temp\\" + os.path.split(current_img_path)[-1].replace('.jpg', '.png')
                    print(current_img_path)
                    im.save(current_img_path)
                # Show another window for the 2D frontal image
                window_main.FindElement('_IMG_PATH_DISPLAY_').Update(values['_IMG_PATH_'])
                imageViewer = ImageViewer(filename_input=current_img_path)
                # imageViewer_active = True
                # window_main.Hide()
                imageViewer.run()
                # imageViewer_active = False
                # window_main.UnHide()
            except Exception as err:
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] +"\n" + errmsg[100:]
                window_main.FindElement('_ERROR_MSG_').Update(errmsg)
        elif event == 'Generate':
            try:
                current_obj_path = values['_OBJ_PATH_'] if values['_OBJ_PATH_'] != '' else current_obj_path
                window_main.FindElement('_OBJ_PATH_DISPLAY_').Update(current_obj_path)
                ### RUN_HERE
                if os.path.isfile(current_obj_path):
                    window = MeshViewer(current_obj_path)
                    pyglet.clock.schedule(window.update)
                    pyglet.app.run()
                    window.show_window()
            except Exception as err:
                errmsg = str(err)
                if len(errmsg) > 100:
                    errmsg = errmsg[:100] + "\n" + errmsg[100:]
                window_main.FindElement('_ERROR_MSG_').Update(errmsg)
    window_main.Close()
