# Automatic 3D Head Modeling From A Few Images

Cheung Yik Kin, Koo Tin Lok , Or Ka Po - Hong Kong University of Science and Technology

ELEC4900 SB3-18 | COMP4901 MXJ3

## Version log
NA

## Dependencies

* Python==3.6.8
* numpy
* opencv-contrib-python
* tensorflow-gpu==1.10.0
* scipy
* scikit-image
* mtcnn
* pyglet
* PyWavefront
* Pillow
* PySimpleGUI

```
pip install -r requirment.txt
```

## Environement setting

##### Symbolic link of Blender file

Windows: 
```
mklink /D <BLENDER FOLDER> Blender
```
Mac/Linux:
``` 
sudo ln -s <BLENDER FOLDER> Blender
```

##### Data

**Hair style:** 

Download [link](http://www-scf.usc.edu/~liwenhu/SHM/database.html), unzip and put /hairstyles/hairstyles into ./Data/hair

**PRNet:**

Download [link](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing), unzip into ./PRNet/Data/net-data)

## Usage

put input image (.jpg) under .\Data\input

```
python main.py
```
input only the filename (e.g. 0.jpg)

## Display UI

-

