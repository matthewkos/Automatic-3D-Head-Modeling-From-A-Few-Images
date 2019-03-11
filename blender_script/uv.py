import bpy
import numpy as np
from os import getcwd as currdir

def adduv(file):
    bpy.context.scene.objects.active = bpy.data.objects['Head']
    img = bpy.ops.image.open(filepath=currdir()+r"//"+file)
    bpy.data.images[file].name = "uv_map"
    return bpy.data.images.get('uv_map')


def apply_uv(img):
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
    #bpy.ops.cycles.use_shading_nodes()
    matnodes = mat.node_tree.nodes
    texture=matnodes.new("ShaderNodeTexImage")
    texture.image = img
    disp = matnodes['Diffuse BSDF'].inputs[0]
    mat.node_tree.links.new(disp, texture.outputs[0])
    
def apply(img):
    head = bpy.data.objects['Head']
    bpy.context.scene.objects.active = head
    for uv_face in head.data.uv_textures.active.data:
        uv_face.image = img
        
    bpy.ops.object.bake_image()


if __name__=="__main__":
    file = r'texture_skin.png'

    img = adduv(file)
    apply_uv(img)