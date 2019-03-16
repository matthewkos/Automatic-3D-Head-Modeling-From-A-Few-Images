import bpy
import bmesh
import time
from os import chdir, getcwd
from os.path import exists, join
import os
from collections.abc import Iterable
import numpy as np
from math import radians
import struct
import json

"""
Basic
"""


def view_selected_index():
    obj = bpy.context.object
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [v for v in bm.verts if v.select]
    return verts


def select(name):
    try:
        obj = bpy.data.objects[name]
        bpy.context.scene.objects.active = obj
        for item in bpy.context.copy()['selected_editable_objects']:
            item.select = False
        obj.select = True
        return obj
    except:
        return None


def getScene():
    return bpy.data.scenes['Scene']


def import_obj(file_loc, name):
    """
    :param file_loc:  file_path
    :param name: as name in context
    :return:
    """
    scene = bpy.context.scene
    scene.objects.active = None
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    bpy.context.selected_objects[0].name = name
    return


def toabscoord(v, mat):
    """
    :param v:  vector
    :param mat: matrix_world
    :return:
    """
    if isinstance(v, Iterable):
        l = []
        for i in v:
            l.append(mat * i)
        return type(v)(l)
    else:
        return mat * v


def objectMode(obj):
    select(obj)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def editMode(obj):
    select(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    return obj


def edge_fit(face, head, mesh, fore_ind, jaw_ind, ind_bound, kpt_ind):
    # fit forehead
    # mask forehead edge between indices [0,480]
    FACE_COUNT = face.shape[0]
    face_top_ind = [i for i in ind_bound if i < 480]  # get indices of forehead on mask
    face_top = face[face_top_ind]  # then obtian their coordinates, middle is no. 13

    neighbour_top = []  # index of nearneibour of forehead on top side of the mask

    for i in fore_ind:
        temp = head[i - FACE_COUNT]  # get one point on fore head (head)
        # get distance from each fore head vertex to mask boundary
        dist = np.square(face_top[:, 0] - temp[0]) + np.square(face_top[:, 1] - temp[1]) + np.square(
            face_top[:, 2] - temp[2])
        neighbour_top.append(face_top_ind[int(np.argmin(dist))])  # store index
    neigh = face[neighbour_top]  # obtain coordinate
    fore = head[fore_ind - FACE_COUNT]
    # top_y = np.mean(neigh[:,1],axis = 0) - np.mean(fore[:,1],axis = 0)
    top_z = np.mean(neigh[:, 2], axis=0) - np.mean(fore[:, 2], axis=0)

    face[:, 2] -= top_z  # align faace using forehead (working)
    mesh = update_mesh(face, head, mesh)
    # ! alignment done

    neigh = face[neighbour_top]
    top_y = neigh[12, 1] - head[fore_ind[12] - FACE_COUNT, 1]
    # top_z = neigh[12,2] - head[fore_ind[12] - FACE_COUNT,2]

    bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(fore_ind, mesh)
    bpy.ops.transform.translate(value=(0, 0, top_y), constraint_axis=(False, False, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')
    # ! jaw part

    # jaw_ind = np.loadtxt(os.path.join(DIR_KPTS, 'jaw.txt').astype(np.int32)
    # bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(jaw_ind, mesh)
    jaw = head[jaw_ind - FACE_COUNT]
    temp2 = face[kpt_ind[8]]

    bottom_y = temp2[1] - np.mean(jaw[:, 1], axis=0)
    bottom_z = temp2[2] - np.mean(jaw[:, 2], axis=0)
    bpy.ops.transform.translate(value=(0, -bottom_z, -bottom_y), constraint_axis=(False, False, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')
    return mesh


def get_kpts():
    # ! load the vertex index correspond to facial landmarks
    # procedure refer to get_kpt_ind.py
    # global kpt_ind, left_ind, fore_ind, jaw_ind, ind_bound, neck_ind
    # DIR_KPTS = '.\\Data\\geometry'
    DIR_KPTS = 'C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\geometry'
    ind_bound = np.loadtxt(os.path.join(DIR_KPTS, 'bound.txt')).astype(np.int32)
    kpt_ind = np.loadtxt(os.path.join(DIR_KPTS, 'kpt_ind.txt')).astype(np.int32)  # ntri x 3
    left_ind = np.loadtxt(os.path.join(DIR_KPTS, 'ear.txt')).astype(np.int32)
    fore_ind = np.loadtxt(os.path.join(DIR_KPTS, 'fore_ind.txt')).astype(np.int32)
    jaw_ind = np.loadtxt(os.path.join(DIR_KPTS, 'jaw.txt')).astype(np.int32)
    neck_ind = np.loadtxt(os.path.join(DIR_KPTS, 'neck.txt')).astype(np.int32)
    return kpt_ind, left_ind, fore_ind, jaw_ind, ind_bound, neck_ind


def getScale(face, head):
    P1_REF = 52447
    P2_REF = 44683
    P21_REF = 28003
    P22_REF = 27792
    FACE_COUNT = face.shape[0]
    p1 = head[P1_REF - FACE_COUNT]
    p2 = head[P2_REF - FACE_COUNT]
    dis1 = abs(p1[0] - p2[0])
    p21 = face[P21_REF]
    p22 = face[P22_REF]
    dis2 = abs(p21[0] - p22[0])
    scal = dis2 / dis1
    return scal


def getPos(face, head, left_ind, kpt_ind):
    x = 0
    FACE_COUNT = face.shape[0]
    left = head[left_ind - FACE_COUNT]
    left_kpt = face[kpt_ind[14:17]]
    y = head[51673 - FACE_COUNT, 1] - face[kpt_ind[8], 1]
    z = np.mean(left[:, 2], axis=0) - np.mean(left_kpt[:, 2], axis=0)
    return x, y, z


def get_meshes(FACE_COUNT):
    """
    get
        the mesrged object,
        meshes of obejct
        Face vertices
        Head vertices
    :return: ob, mesh,f ace, head
    """
    # global ob, mesh, face, head
    ob = bpy.data.objects['Object']
    bpy.context.scene.objects.active = ob  # ! required to select correct context
    # bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.object.mode_set(mode='EDIT')
    mesh = bmesh.from_edit_mesh(ob.data)
    mesh.verts.ensure_lookup_table()
    bpy.ops.mesh.select_all(action='DESELECT')
    coord = [x.co for x in mesh.verts]
    face = np.array(coord[:FACE_COUNT])  # coordinates of face
    head = np.array(coord[FACE_COUNT:])  # coordinates of face
    return ob, mesh, face, head


def update_mesh(face_new, head_new, mesh):
    """
    WARNING: FACE_COUNT may vary
    :param face_new:
    :param head_new:
    :param mesh:
    :return:
    """
    FACE_COUNT = face_new.shape[0]
    for i in range(FACE_COUNT):
        mesh.verts[i].co = face_new[i, :]
    #
    # for ii in range(FACE_COUNT, 59393):
    #     mesh.verts[i].co = face_new[i, :]
    return mesh


def import_face(file_loc='C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\mask\\0.obj'):
    import_obj(file_loc, 'Object')
    bpy.context.scene.objects.active = None


# def import_head():
#     file_loc = 'C:\\Users\\KTL\\Desktop\\FYP-code\\split\\head_dissolve.obj'  # TODO change to more recent head model
#
#     import_obj(file_loc, 'head')
#     select('head')
#     bpy.context.scene.objects.active.scale = (1.165, 1.165, 1.165)
#     bpy.context.scene.objects.active = None


def face_join():
    """
    Join the objects into one mesh
    :return:
    """
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH':
            obs.append(ob)
    ctx = bpy.context.copy()
    # one of the objects to join
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    # we need the scene bases as well for joining
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)


def sel_vert(ind, mesh):
    ob = bpy.data.objects['Object']
    bpy.context.scene.objects.active = ob  # select desired object first
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    mesh.verts.ensure_lookup_table()

    for i in ind:
        mesh.verts[i].select = True

    # bpy.context.scene.objects.active = ob # refresh the scene


def align_face(MASK_DATA='0.obj'):
    """
    GET data from files
    """
    FACE_COUNT = 43866  # the number of vertex in face mask
    # DIR_MASK = 'C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\mask'
    if not os.path.exists(".\\config.ini"):
        raise FileNotFoundError("config.ini")
    with open('.\\config.ini', 'r') as json_file:
        json_data = json.load(json_file)
        for (k, v) in json_data.items():
            exec("{} = {}".format(k, v))
        print('wer')

    kpt_ind, left_ind, fore_ind, jaw_ind, ind_bound, neck_ind = get_kpts()
    objectMode('Head')
    import_obj(os.path.join(DIR_MASK, MASK_DATA), 'Object')
    bpy.context.scene.objects.active = None
    """
    Merge
    """
    face_join()
    ob, mesh, face, head = get_meshes(FACE_COUNT)

    """
    Scale face
    """
    scal = getScale(face, head)
    face = face / scal
    """
    Translate face
    """
    transx, transy, transz = getPos(face, head, left_ind, kpt_ind)
    # TODO: use blender. move
    face[:, 0] += 0
    face[:, 1] += transy
    face[:, 2] += transz
    mesh = update_mesh(face, head, mesh)

    """
    EDGE_FIT
    """
    edge_fit(face, head, mesh, fore_ind, jaw_ind, ind_bound, kpt_ind)

    bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(neck_ind, mesh)
    bpy.ops.mesh.delete(type='VERT')
    # sel_vert(ear_ind, mesh)
    # bpy.ops.mesh.delete(type='VERT')

    bpy.ops.mesh.separate(type='MATERIAL')

    bpy.data.objects['Object'].name = 'Head'
    bpy.data.meshes['0'].name = 'Head-mesh'
    bpy.data.objects['Object.001'].name = 'Face'
    bpy.data.meshes['0.001'].name = 'Face-mesh'


if __name__ == '__main__':
    """
    import var from config.ini
    """
    if not os.path.exists(".\\config.ini"):
        raise FileNotFoundError("config.ini")
    with open('.\\config.ini', 'r') as json_file:
        json_data = json.load(json_file)
        for (k, v) in json_data.items():
            exec("{} = {}".format(k, v))
    align_face(MASK_DATA=MASK_DATA)
