import bpy
import bmesh
from os import chdir, getcwd
from os.path import exists, join
import os
from collections.abc import Iterable
import numpy as np
from math import radians
import struct
import json
import argparse

"""
DEFINE CONSTANT
"""

"""
Basic
"""


def view_selected_index():
    """
    The index of selected vertices in 3D view
    :return: a list of verts
    """
    obj = bpy.context.object
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [v for v in bm.verts if v.select]
    return verts


def select(name):
    """
    Select the Object and the object only by name in:
        3D view, context
    :param name: name of object (as in context)
    :return: the bpy data objects
    """
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
    """
    API to get the Current Scene
    :return: Scene
    """

    return bpy.data.scenes['Scene']


def import_obj(file_loc, name):
    """\
    :param file_loc:  file_path
    :param name: as name in context
    :return: the imported object
    """
    scene = bpy.context.scene
    scene.objects.active = None
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    bpy.context.selected_objects[0].name = name
    return select(name)


def toabscoord(v, mat):
    """
    Change the local vertices coordinate to absolute coordinate
    :param v:  vector or a list of vertices
    :param mat: matrix_world
    :return: the Absolute vertices
    """
    if isinstance(v, Iterable):
        l = []
        for i in v:
            l.append(mat * i)
        return type(v)(l)
    else:
        return mat * v


def objectMode(obj):
    """
    Select the object and change to object mode
    :param obj: string, name of object
    :return: bpy data object
    """
    o = select(obj)
    bpy.ops.object.mode_set(mode='OBJECT')
    return o


def editMode(obj):
    """
    Select the object and change to edit mode
    :param obj: string, name of object
    :return: bpy data object
    """
    o = select(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    return o


"""
Transform
"""


def scale_x(ratio):
    bpy.ops.transform.resize(value=(ratio, 1, 1), constraint_axis=(True, False, False), constraint_orientation='GLOBAL',
                             mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH',
                             proportional_size=1)


def scale(ratio_x, ratio_y=None, ratio_z=None):
    if ratio_y is None:
        ratio_y = ratio_x
    if ratio_z is None:
        ratio_z = ratio_x
    bpy.ops.transform.resize(value=(ratio_x, ratio_y, ratio_z), constraint_axis=(True, True, True),
                             constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)


def move(x, y, z):
    bpy.ops.transform.translate(value=(x, y, z), constraint_axis=(False, False, False), constraint_orientation='GLOBAL',
                                mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH',
                                proportional_size=1)


def rotate(deg, axis):
    a = [0, 0, 0]
    a[axis] = 1
    bpy.ops.transform.rotate(value=radians(deg), axis=a,
                             constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False,
                             proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    return


"""
Face
"""


def modify_face():
    # bpy.context.space_data.context = 'MODIFIER'
    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].ratio = 0.3
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")
    bpy.ops.object.modifier_add(type='SMOOTH')
    bpy.context.object.modifiers["Smooth"].iterations = 2
    bpy.context.object.modifiers["Smooth"].factor = 1.51
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Smooth")
    bpy.ops.object.shade_smooth()
    return


"""
Head
"""


def applyHeadSub():
    objectMode('Head')
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Subsurf")
    return


def getUpperHead():
    head = select('Head')
    bm = bmesh.new()
    bm.from_object(head, getScene())
    bmv = list(bm.verts)
    ref_point_z = bmv[10507]
    ref_point_y = bmv[17869]
    upperHead = [v.co.copy() for v in bmv if v.co.z >= ref_point_z.co.z and v.co.y >= ref_point_y.co.y]
    return upperHead


def getFaceHeadV():
    face = select('Face')
    fbm = bmesh.new()
    fbm.from_object(face, getScene())
    fbmv = fbm.verts
    fbmvl = list(fbmv)
    fbmvl = sorted(fbmvl, key=lambda v: (v.co.x, v.co.y, v.co.z))

    head = select('Head')
    hbm = bmesh.new()
    hbm.from_object(head, getScene())
    hbmv = hbm.verts
    hbmvl = list(hbmv)
    hbmvl = sorted(hbmvl, key=lambda v: (v.co.x, v.co.y, v.co.z))

    f1, f2, h1, h2 = fbmvl[0].co.copy(), fbmvl[-1].co.copy(), hbmvl[0].co.copy(), hbmvl[-1].co.copy()
    fbm.free()
    hbm.free()

    return (f1, f2, h1, h2), (face.matrix_world, head.matrix_world)


def headUV(file):
    if bpy.data.images.get('uv_map') is not None:
        bpy.data.images.remove(bpy.data.images['uv_map'])
    bpy.context.scene.objects.active = bpy.data.objects['Head']
    img = bpy.ops.image.open(filepath=join(getcwd(),file))
    bpy.data.images[os.path.split(file)[-1]].name = "uv_map"
    img = bpy.data.images.get('uv_map')
    bpy.context.scene.objects.active = bpy.data.objects['Head']
    # Get material
    mat = bpy.data.materials.get("Material")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")
    # Assign it to object
    if bpy.data.objects['Head'].data.materials:
        # assign to 1st material slot
        bpy.data.objects['Head'].data.materials[0] = mat
    else:
        # no slots
        bpy.data.objects['Head'].data.materials.append(mat)
    mat.use_nodes = True
    # bpy.ops.cycles.use_shading_nodes()
    matnodes = mat.node_tree.nodes
    texture = matnodes.new("ShaderNodeTexImage")
    texture.image = img
    disp = matnodes['Diffuse BSDF'].inputs[0]
    mat.node_tree.links.new(disp, texture.outputs[0])
    return


"""
Hair
"""


def decode_as_int(b):
    return struct.unpack("<L", b)[0]


def decode_as_3f(b):
    return [struct.unpack("<f", b[0:4])[0], struct.unpack("<f", b[4:8])[0], struct.unpack("<f", b[8:12])[0]]


def read_data(file_name):
    with open(file_name, 'rb') as f:
        num_strend = decode_as_int(f.read(4))
        S = []
        for i in range(num_strend):
            num_vertices = decode_as_int(f.read(4))
            v = []
            for j in range(num_vertices):
                v.append(decode_as_3f(f.read(12)))
            if len(v) > 1:
                S.append(v)
    return np.array(S)


def gen_strend(S):
    mesh = bpy.data.meshes.new("mesh")
    obj = bpy.data.objects.new("Hair", mesh)
    # obj.parent = bpy.data.objects['Hair']
    # obj = bpy.data.objects['Hair']
    scene = bpy.context.scene
    scene.objects.link(obj)
    scene.objects.active = obj
    obj.select = True
    mesh = bpy.context.object.data
    bm = bmesh.new()
    for ns, verts in enumerate(S):
        bvp0 = None
        bvp1 = None
        for nv, v in enumerate(verts):
            bv = bm.verts.new(v)
            bv.index = ns * 100 + nv
            # bv1 = bm.verts.new(v)
            # bv.index = nv + ns * 100
            if bvp1 is not None:
                bf = bm.faces.new([bv, bvp0, bvp1])
                bf.index = nv + ns * 100
            bvp1 = bvp0
            bvp0 = bv
    bm.to_mesh(mesh)
    bm.free()


def gen_strends(S):
    for ns, s in enumerate(S):
        gen_strend(s, ns)


def genHair(file_name):
    if exists(file_name):
        S = read_data(file_name)
        gen_strend(S)
        select('Hair')
        rotate(90, 0)
    else:
        raise Exception("file not found")
    return


def import_test_head():
    scene = bpy.context.scene
    scene.objects.active = None
    file_loc = OBJ_HEAD_MODEL_HAIR
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    bpy.context.selected_objects[0].name = "Head"
    bpy.ops.transform.rotate(value=-1.5708, axis=(1, 0, 0), constraint_axis=(True, False, False),
                             constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)
    return


def getRoot():
    hair = select('Hair')
    bm = bmesh.new()
    bm.from_object(hair, getScene())
    vl = list(bm.verts)
    roots = [v.co.copy() for v in vl if v.index % 100 == 0]
    return roots


"""
Integration
"""


def AlignHeadHair():
    hair = select('Hair')
    head = select('Head')
    root = getRoot()
    upper = getUpperHead()
    root = toabscoord(root, hair.matrix_world)
    upper = toabscoord(upper, head.matrix_world)
    root_x = sorted(root, key=lambda v: (v.x, v.y, v.z))
    upper_x = sorted(upper, key=lambda v: (v.x, v.y, v.z))
    root_y = sorted(root, key=lambda v: (v.y, v.x, v.z))
    upper_y = sorted(upper, key=lambda v: (v.y, v.x, v.z))

    # head / hair
    n1, n2 = upper_x[0], upper_x[-1]
    d1, d2 = root_x[0], root_x[-1]
    n3, n4 = upper_y[0], upper_y[-1]
    d3, d4 = root_y[0], root_y[-1]

    ratio_x = (n2.x - n1.x) / (d2.x - d1.x)
    ratio_y = (n4.y - n3.y) / (d4.y - d3.y)
    select('Hair')
    scale(ratio_x, ratio_y, ratio_x)

    hair = select('Hair')
    head = select('Head')
    root = getRoot()
    upper = getUpperHead()
    root = toabscoord(root, hair.matrix_world)
    upper = toabscoord(upper, head.matrix_world)
    root_x = sorted(root, key=lambda v: (v.x, v.y, v.z))
    upper_x = sorted(upper, key=lambda v: (v.x, v.y, v.z))
    root_y = sorted(root, key=lambda v: (v.y, v.x, v.z))
    upper_y = sorted(upper, key=lambda v: (v.y, v.x, v.z))

    n1, n2 = upper_x[0], upper_x[-1]
    d1, d2 = root_x[0], root_x[-1]
    n3, n4 = upper_y[0], upper_y[-1]
    d3, d4 = root_y[0], root_y[-1]
    diff = ((n1 - d1) + (n2 - d2) + (n3 - d3) + (n4 - d4)) / 4
    objectMode('Hair')
    move(*list(diff))
    move(0, 0, -0.2)
    scale(1.1,1.1,1.1)
    move(0,0,-50)
    return


def modiftHair():
    # select('Hair')
    editMode('Hair')
    bpy.ops.mesh.select_all(action='TOGGLE')
    bpy.ops.mesh.remove_doubles()
    objectMode('Hair')

    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].ratio = 0.5
    bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")

    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness_clamp = 2
    bpy.context.object.modifiers["Solidify"].offset = 0
    bpy.context.object.modifiers["Solidify"].thickness = 1  # 0.001-0.003
    bpy.context.object.modifiers["Solidify"].use_rim = False
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Solidify")

    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].ratio = 0.5
    bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")

    return


def colorHair():
    select('Hair')
    bpy.context.scene.objects.active = bpy.data.objects['Hair']
    # Get material
    mat = bpy.data.materials.get("Material")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")

    # Assign it to object
    if bpy.data.objects['Hair'].data.materials:
        # assign to 1st material slot
        bpy.data.objects['Hair'].data.materials[0] = mat
    else:
        # no slots
        bpy.data.objects['Hair'].data.materials.append(mat)
    mat.use_nodes = True
    # bpy.ops.cycles.use_shading_nodes()
    matnodes = mat.node_tree.nodes
    matnodes.remove(matnodes['Diffuse BSDF'])

    texture = matnodes.new("ShaderNodeBsdfHair")
    matnodes["Hair BSDF"].inputs[0].default_value = (0, 0, 0, 1)
    mat.node_tree.links.new(matnodes["Material Output"].inputs[0], texture.outputs[0])
    mat.node_tree.links.new(matnodes["Material Output"].inputs[1], texture.outputs[0])
    mat.node_tree.links.new(matnodes["Material Output"].inputs[2], texture.outputs[0])
    return


def removeHair():
    hair = select('Hair')
    if hair is not None:
        bpy.ops.object.delete()
    return


def importHair(file):
    if exists(file):
        if select('Hair') is None:
            genHair(file)
            hair = select('Hair')
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    else:
        raise Exception('File not found')

def output(file_name):
    bpy.ops.export_scene.obj(filepath=file_name, check_existing=False, keep_vertex_order=True)


if __name__ == '__main__':

    with open('.\config.ini', 'r') as json_file:
        json_data = json.load(json_file)
        for (k, v) in json_data.items():
            exec("{} = {}".format(k, v))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("texture", help="texture file name (,jpg/,png)")
    # parser.add_argument("mask", help="mask file name (.obj)")
    # parser.add_argument("hair", help="hair file name (.data)")
    # parser.add_argument("out", help="output file name (,obj)")
    # parser.add_argument("--hair-test", default="Data\\hair\\head_model.obj", help="")
    # parser.add_argument("--dir-texture",'-t', default="Data\\texture", help="")
    # parser.add_argument("--dir-mask", '-m', default="Data\\mask", help="")
    # parser.add_argument("--dir-hair", '-H', default="Data\\hair", help="")
    # parser.add_argument("--dir-out", '-o', default="output", help="")
    #
    # args = parser.parse_args()
    # OBJ_HEAD_MODEL_HAIR = args.hair_test
    # TEXTURE_DATA = args.texture
    # MASK_DATA = args.mask
    # HAIR_DATA = args.hair
    # OUT_DATA = args.out
    # DIR_TEXTURE = args.dir_texture
    # DIR_MASK = args.dir_mask
    # DIR_HAIR = args.dir_hair
    # DIR_OUT = args.dir_out

    """
    Head
    """
    applyHeadSub()
    headUV(join(DIR_TEXTURE, TEXTURE_DATA))

    """
    Face
    """
    if select('Face') is None:
        import_obj(file_loc=join(DIR_MASK, MASK_DATA), name='Face')
        select('Face')
        modify_face()

    # change ratio
    (f1, f2, h1, h2), (fmat, hmat) = getFaceHeadV()
    f1, f2 = toabscoord([f1, f2], fmat)
    h1, h2 = toabscoord([h1, h2], hmat)
    ratio = (f2.x - f1.x) / (h2.x - h1.x)
    objectMode('Head')
    scale(ratio)

    # translate
    (f1, f2, h1, h2), (fmat, hmat) = getFaceHeadV()
    f1, f2 = toabscoord([f1, f2], fmat)
    h1, h2 = toabscoord([h1, h2], hmat)
    diff = ((f1 - h1) + (f2 - h2)) / 2
    objectMode('Head')
    move(*list(diff))

    # from input photo

    # hair
    removeHair()
    importHair(join(DIR_HAIR, HAIR_DATA))
    AlignHeadHair()
    modiftHair()
    # colorHair()
    objectMode('Hair')
    output(join(DIR_OUT,OUT_DATA))
