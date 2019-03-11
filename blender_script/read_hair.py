import struct
import numpy as np
import bpy
import bmesh
from os.path import exists


def decode_as_int(b):
    return struct.unpack("<L", b)[0]


def decode_as_3f(b):
    return [struct.unpack("<f", b[0:4])[0], struct.unpack("<f", b[4:8])[0], struct.unpack("<f", b[8:12])[0]]


def read_data(file_name):
    with open(file_name, 'rb') as f:
        num_strend = decode_as_int(f.read(4))
        print(num_strend)
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
            # bv1 = bm.verts.new(v)
            # bv.index = nv + ns * 100
            if bvp1 is not None:
                bf = bm.faces.new([bv, bvp0, bvp1])
                bf.index = nv + ns * 100
            bvp1 = bvp0
            bvp0 = bv

    bm.to_mesh(mesh)
    bm.free()
    # bpy.ops.transform.rotate(value=1.5708, axis=(1, 0, 0), constraint_axis=(True, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)


def gen_strends(S):
    for ns, s in enumerate(S):
        gen_strend(s, ns)


def __run__(file_name):
    S = read_data(file_name)
    # print(S)
    gen_strend(S)
    a = (0, 0.88794, -0.15206)


def genHair(file_name):
    if exists(file_name):
        S = read_data(file_name)
        gem_strend(S)
    else:
        raise Exception("file not found")
    return


def import_obj():
    scene = bpy.context.scene
    scene.objects.active = None
    file_loc = r'D:\Cloud\OneDrive - HKUST Connect\HKUST\FYP\Code\Hair\hairstyles\hairstyles\head_model.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    bpy.context.selected_objects[0].name = "Head"
    bpy.ops.transform.rotate(value=-1.5708, axis=(1, 0, 0), constraint_axis=(True, False, False),
                             constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)


if __name__ == '__main__':
    name = "strands00001.data"
    # import_obj()
    __run__(name)
