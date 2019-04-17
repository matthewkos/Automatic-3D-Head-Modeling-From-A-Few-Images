import bpy
import bmesh
import numpy as np


# TODO Use Mask modifier to hide in object mode, or delete vertices
# ! NOTE y-z coordinates of transform and verteices are inverted
# python demo.py --isDlib True --isFront True --isTexture True --isMask True
def Init():
    # global face
    get_kpts()
    found = 'Object' in bpy.data.objects
    if not (found):
        import_head()
        import_obj()  # import face model
        face_join()
        get_mesh()
        scal = get_scale()
        face_new = face / scal
        update_mesh(face_new, head)

        get_mesh()
        update_mesh(face, head)
        transx, transy, transz = get_pos()

        face[:, 0] += 0
        face[:, 1] += transy
        face[:, 2] += transz
        update_mesh(face, head)
        get_mesh()
        scal = get_scale()
        face_new = face / scal

        update_mesh(face_new, head)
        hide()
        edge_fit()
    # get_mesh()


def edge_fit():
    global fore_ind, jaw_ind, ind_bound, face_top_ind

    # ! fit forehead
    # mask forehead edge between indices [0,480]
    face_top_ind = [i for i in ind_bound if i < 480]  # get indices of forehead on mask
    face_top = face[face_top_ind]  # then obtian their coordinates, middle is no. 13

    neighbour_top = []  # index of nearneibour of forehead on top side of the mask
    neigh = []
    for i in fore_ind:
        get_mesh()
        temp = head[i - face_count]  # get one point on fore head (head)
        # get distance from each fore head vertex to mask boundary
        dist = np.square(face_top[:, 0] - temp[0]) + np.square(face_top[:, 1] - temp[1]) + np.square(
            face_top[:, 2] - temp[2])
        neighbour_top.append(face_top_ind[np.argmin(dist)])  # store index

    neigh = face[neighbour_top]  # obtain coordinate

    fore = head[fore_ind - face_count]
    # top_y = np.mean(neigh[:,1],axis = 0) - np.mean(fore[:,1],axis = 0)
    top_z = np.mean(neigh[:, 2], axis=0) - np.mean(fore[:, 2], axis=0)

    face[:, 2] -= top_z  # align faace using forehead (working)
    update_mesh(face, head)
    get_mesh()

    neigh = face[neighbour_top]
    top_y = neigh[12, 1] - head[fore_ind[12] - face_count, 1]
    # top_z = neigh[12,2] - head[fore_ind[12] - face_count,2]

    bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(fore_ind, False, [])
    bpy.ops.transform.translate(value=(0, 0, top_y), constraint_axis=(False, False, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')
    # ! jaw part

    jaw_ind = np.loadtxt('/home/ykcheungab/fyp_ws/data/jaw.txt').astype(np.int32)
    # bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(jaw_ind, False, [])
    jaw = head[jaw_ind - face_count]
    temp2 = face[kpt_ind[8]]

    bottom_y = temp2[1] - np.mean(jaw[:, 1], axis=0)
    bottom_z = temp2[2] - np.mean(jaw[:, 2], axis=0)
    bpy.ops.transform.translate(value=(0, -bottom_z, -bottom_y), constraint_axis=(False, False, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')


def hide():
    hide_ind = np.loadtxt('/home/ykcheungab/fyp_ws/data/hide_face.txt').astype(np.int32)
    # hide_ind = [54641]
    # print(hide_ind)
    # hide_ind = np.concatenate((hide_ind , [54641]),axis = 0)
    bpy.ops.mesh.select_all(action='DESELECT')

    sel_vert(hide_ind, False, [])
    bpy.ops.mesh.hide(unselected=False)

    # bpy.ops.object.modifier_add(type='MASK')
    # bpy.context.object.modifiers["Mask"].vertex_group = "HIDE"
    # bpy.context.object.modifiers["Mask"].invert_vertex_group = True
    bpy.ops.mesh.select_all(action='DESELECT')


def get_kpts():
    # ! load the vertex index correspond to facial landmarks
    # procedure refer to get_kpt_ind.py
    global kpt_ind, left_ind, fore_ind, jaw_ind
    kpt_ind = np.loadtxt('/home/ykcheungab/fyp_ws/data/kpt_ind.txt').astype(np.int32)  # ntri x 3
    left_ind = np.loadtxt('/home/ykcheungab/fyp_ws/data/ear.txt').astype(np.int32)
    fore_ind = np.loadtxt('./data/fore_ind.txt').astype(np.int32)
    jaw_ind = np.loadtxt('/home/ykcheungab/fyp_ws/data/jaw.txt').astype(np.int32)


def get_scale():
    p1 = head[52447 - face_count]
    p2 = head[44683 - face_count]
    dis1 = abs(p1[0] - p2[0])
    # print(dis1)

    p21 = face[28003]
    p22 = face[27792]
    dis2 = abs(p21[0] - p22[0])
    # print(dis2)

    scal = dis2 / dis1
    return scal


def get_pos():
    x = 0
    y = 0
    z = 0

    left = head[left_ind - face_count]
    left_kpt = face[kpt_ind[14:17]]

    y = head[51673 - face_count, 1] - face[kpt_ind[8], 1]
    z = np.mean(left[:, 2], axis=0) - np.mean(left_kpt[:, 2], axis=0)

    return (x, y, z)


def get_mesh():
    global ob, mesh, face, head
    ob = bpy.data.objects['Object']
    bpy.context.scene.objects.active = ob  # ! required to select correct context
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='EDIT')  # ! not sure why need two times
    mesh = bmesh.from_edit_mesh(ob.data)
    mesh.verts.ensure_lookup_table()
    bpy.ops.mesh.select_all(action='DESELECT')

    face = []
    for i in range(face_count):
        face.append(mesh.verts[i].co)

    face = np.array(face)  # coordinates of face
    head = []
    for i in range(face_count, 59393):
        head.append(mesh.verts[i].co)
    head = np.array(head)  # coordinates of face
    # print(len(head))


def update_mesh(face_new, head_new):
    for i in range(face_count):
        mesh.verts[i].co = face_new[i, :]

    for ii in range(face_count, 59393):
        mesh.verts[i].co = face_new[i, :]


def import_obj():
    global ob, mesh
    bpy.context.scene.objects.active = None
    # check the existence of  'Object'
    found = default in bpy.data.objects
    # import .obj file to blender and named "Object"
    if not (found):
        file_loc = '/home/ykcheungab/fyp_ws/model/0.obj'
        imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
        bpy.context.selected_objects[0].name = default
        bpy.context.selected_objects[0].data.name = default
        # ! outdated due to mask is manipulated after joining
    # ob   = bpy.data.objects['Object']
    # bpy.context.scene.objects.active = ob # required to select correct context
    # bpy.ops.object.mode_set(mode = 'EDIT')
    # bpy.ops.object.mode_set(mode = 'EDIT') # not sure why need two times
    # mesh = bmesh.from_edit_mesh(ob.data)
    # Do not change to OBJECT mode, otherwise mesh data will be cleared    
    # bpy.context.object.active_material.emit = 1
    # bpy.ops.mesh.select_all(action='DESELECT')


def import_head():
    global hd, mesh_hd
    name = 'head'
    bpy.context.scene.objects.active = None
    # check the existence of  'Object'
    found = name in bpy.data.objects
    # import .obj file to blender and named "head"
    if not (found):
        file_loc = '/home/ykcheungab/fyp_ws/model/head_dissolve.obj'  # TODO change to more recent head model
        imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
        bpy.context.selected_objects[0].name = name
        bpy.context.selected_objects[0].data.name = name
        hd = bpy.data.objects[name]
        bpy.context.scene.objects.active = hd  # ! required to select correct context
        # ! scale obtained by trial and visual inspection, dont't use more difficult resize transform
        bpy.context.scene.objects.active.scale = (1.165, 1.165, 1.165)
    # ! outdated due to head is manipulated after joining
    # hd   = bpy.data.objects[name]
    # bpy.context.scene.objects.active = hd #! required to select correct context
    # bpy.ops.object.mode_set(mode = 'EDIT')
    # bpy.ops.object.mode_set(mode = 'EDIT') # not sure why need run two times to get into edit mode
    # mesh_hd = bmesh.from_edit_mesh(hd.data)
    # bpy.ops.mesh.select_all(action='DESELECT')


def face_join():
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
    # bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.join(ctx)


def resize_head():
    head = bpy.data.objects['Cube']
    bpy.context.scene.objects.active = head
    # bpy.ops.transform.resize(value=(1.09186, 1.09186, 1.09186), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    # bpy.ops.transform.resize(value=(1.06226, 1.06226, 1.06226), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.context.scene.objects.active.scale = (1.165, 1.165, 1.165)


def sel_vert(ind, partial, ignore):
    bpy.context.scene.objects.active = ob  # select desired object first
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    mesh.verts.ensure_lookup_table()
    # bpy.ops.mesh.select_all(action='DESELECT')
    if partial:
        for i in ind:
            if not (ignore[0] <= i <= ignore[1]):
                mesh.verts[i].select = True
    else:
        for i in ind:
            mesh.verts[i].select = True
    bpy.context.scene.objects.active = ob  # refresh the scene


print('running')
ind_bound = [0, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53,
             55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105,
             107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
             149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
             191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231,
             233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259, 261, 263, 265, 267, 269, 271, 273,
             275, 277, 279, 281, 283, 285, 287, 289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315,
             317, 319, 321, 323, 325, 327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351, 353, 355, 357,
             359, 361, 363, 365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399,
             401, 403, 405, 407, 409, 411, 413, 415, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439, 441,
             443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 481, 483,
             485, 487, 489, 490, 491, 736, 737, 982, 983, 1228, 1229, 1474, 1475, 1720, 1721, 1966, 1967, 2212, 2213,
             2458, 2459, 2704, 2705, 2950, 2951, 3196, 3197, 3442, 3443, 3688, 3689, 3934, 3935, 4180, 4181, 4426, 4427,
             4672, 4673, 4918, 4919, 5164, 5165, 5410, 5411, 5656, 5657, 5902, 5903, 6148, 6149, 6394, 6395, 6640, 6641,
             6886, 6887, 7132, 7133, 7378, 7379, 7624, 7625, 7870, 7871, 8116, 8117, 8362, 8363, 8608, 8609, 8854, 8855,
             9100, 9101, 9346, 9347, 9592, 9593, 9838, 9839, 10084, 10085, 10330, 10331, 10576, 10577, 10822, 10823,
             11068, 11069, 11314, 11315, 11560, 11561, 11806, 11807, 12052, 12053, 12298, 12299, 12544, 12545, 12790,
             12791, 13036, 13037, 13282, 13283, 13528, 13529, 13774, 13775, 14020, 14021, 14266, 14267, 14512, 14513,
             14758, 14759, 15004, 15005, 15250, 15251, 15496, 15497, 15742, 15743, 15988, 15989, 16234, 16235, 16480,
             16481, 16726, 16727, 16972, 16973, 17218, 17219, 17464, 17465, 17710, 17711, 17956, 17957, 18202, 18203,
             18448, 18449, 18694, 18695, 18940, 18941, 19186, 19187, 19432, 19433, 19678, 19679, 19924, 19925, 20170,
             20171, 20416, 20417, 20662, 20663, 20908, 20909, 21154, 21155, 21400, 21401, 21646, 21647, 21892, 21893,
             22138, 22139, 22384, 22385, 22630, 22631, 22876, 22877, 23122, 23123, 23368, 23369, 23614, 23615, 23860,
             23861, 24106, 24107, 24352, 24353, 24598, 24599, 24844, 24845, 25090, 25091, 25336, 25337, 25582, 25583,
             25828, 25829, 26074, 26075, 26319, 26320, 26321, 26564, 26565, 26808, 26809, 27051, 27052, 27053, 27294,
             27295, 27535, 27536, 27537, 27776, 27777, 28015, 28016, 28017, 28254, 28255, 28492, 28493, 28494, 28729,
             28730, 28965, 28966, 28967, 29200, 29201, 29434, 29435, 29667, 29668, 29669, 29900, 29901, 30132, 30133,
             30134, 30363, 30364, 30593, 30594, 30595, 30822, 30823, 31050, 31051, 31052, 31277, 31278, 31503, 31504,
             31729, 31730, 31731, 31954, 31955, 32178, 32179, 32180, 32401, 32402, 32623, 32624, 32625, 32844, 32845,
             33064, 33065, 33066, 33284, 33285, 33502, 33503, 33504, 33720, 33721, 33936, 33937, 34152, 34153, 34154,
             34367, 34368, 34581, 34582, 34583, 34795, 34796, 35007, 35008, 35009, 35219, 35220, 35429, 35430, 35431,
             35639, 35640, 35847, 35848, 35849, 36055, 36056, 36261, 36262, 36263, 36467, 36468, 36671, 36672, 36673,
             36875, 36876, 37077, 37078, 37079, 37279, 37280, 37479, 37480, 37481, 37679, 37680, 37681, 37877, 37878,
             38073, 38074, 38075, 38269, 38270, 38271, 38463, 38464, 38655, 38656, 38657, 38847, 38848, 39037, 39038,
             39039, 39227, 39228, 39229, 39415, 39416, 39601, 39602, 39603, 39787, 39788, 39789, 39971, 39972, 39973,
             40153, 40154, 40155, 40333, 40334, 40335, 40511, 40512, 40687, 40688, 40689, 40863, 40864, 40865, 41037,
             41038, 41039, 41209, 41210, 41211, 41379, 41380, 41381, 41547, 41548, 41549, 41713, 41714, 41715, 41716,
             41876, 41877, 41878, 41879, 41880, 42036, 42037, 42038, 42039, 42040, 42192, 42193, 42194, 42195, 42196,
             42344, 42345, 42346, 42347, 42348, 42492, 42493, 42494, 42495, 42637, 42638, 42639, 42640, 42778, 42779,
             42780, 42781, 42782, 42916, 42917, 42918, 42919, 42920, 43050, 43051, 43052, 43053, 43054, 43180, 43181,
             43182, 43183, 43184, 43306, 43307, 43308, 43309, 43310, 43311, 43312, 43426, 43427, 43428, 43429, 43430,
             43431, 43432, 43433, 43434, 43540, 43541, 43542, 43543, 43544, 43545, 43546, 43547, 43548, 43549, 43550,
             43644, 43645, 43646, 43647, 43648, 43649, 43650, 43651, 43652, 43653, 43654, 43655, 43656, 43738, 43739,
             43740, 43741, 43742, 43743, 43744, 43745, 43746, 43747, 43748, 43749, 43750, 43751, 43752, 43753, 43754,
             43755, 43756, 43757, 43758, 43759, 43760, 43761, 43762, 43763, 43764, 43765, 43805, 43806, 43807, 43808,
             43809, 43810, 43811, 43812, 43813, 43814, 43815, 43816, 43817, 43818, 43819, 43820, 43821, 43822, 43823,
             43824, 43825, 43826, 43827, 43828, 43829, 43830, 43831, 43832, 43833, 43834, 43835, 43836, 43837, 43838,
             43839, 43840, 43841, 43842, 43843, 43844, 43845, 43846, 43847, 43848, 43849, 43850, 43851, 43852, 43853,
             43854, 43855, 43856, 43857, 43858, 43859, 43860, 43861, 43862, 43863, 43864, 43865]
# jaw_ind = [53378,54763,54765,54766,54767,54768,54769,54770, 51673,51672,51667,51666,51554,51553,51546,51545,45644]
# left_ind = [59371,59372,59367,59368,59375,59376,59369,59370,59377,59378,59373,59374,59379]
# left = [[59383,28003],[59380,33277]]
# fore = [53943,53939,53940,53948,53949,53959,53960,53963,52767,52763,52764,52770,45027,45025,45009,45010,45011,46254,46247,46248,46242,46240,46229,46230,46231]
# get_kpts()
default = 'Object'  # default name for face model
scene = bpy.context.scene
face_count = 43866
Init()

# sel_vert(ind_bound,True,[0,35430]) # exclude ears
# sel_vert(range(43866),False,[]) # select face all


# sel_vert(kpt_ind,False,[])
# el_vert(left_ind,False,[])
# sel_vert(fore_ind,False,[])
# sel_vert([52447,44683],False,[])
# sel_vert(jaw_ind,False,[])
# sel_vert(range(face_count),False,[])
# sel_vert(face_top_ind,False,[])
# bpy.ops.transform.translate(value=(0, 0.126863, 0.0671628), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED', proportional_edit_falloff='SMOOTH', proportional_size=1)
