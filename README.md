# Automatic 3D Head Modeling From A Few Images

Cheung Yik Kin, Koo Tin Lok, Or Ka Po - Hong Kong University of Science and Technology
ELEC4900 SB3-18 | COMP4901 MXJ3

## Version log
-

## Dependencies

* Anaconda
* Python 3.6.8
As state in requirment.txt
* numpy
* opencv-contrib-python
* tensorflow-gpu
* scipy
* skimage
* mtcnn

```bash
pip install -r requirment.txt
```

## Environement setting

######Symbolic link of Blender file

Windows: 
mklink /D <BLENDER FOLDER> Blender
Mac/Linux: 
sudo ln -s <BLENDER FOLDER> Blender

######Data
Hair style: unzip the files from 
to ./Data/hair
PRNet: 