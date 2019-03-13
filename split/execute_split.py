import bpy
import bmesh
import numpy as np 
import time
import os

#TODO Use Mask modifier to hide in object mode, or delete vertices
#! NOTE y-z coordinates of transform and verteices are inverted
# python demo.py --isDlib True --isFront True --isTexture True --isMask True

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
    """\
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
        return mat * i


def objectMode(obj):
    select(obj)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def editMode(obj):
    select(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    return obj


def edge_fit():
    global fore_ind, jaw_ind, ind_bound, face_top_ind

    #! fit forehead
    # mask forehead edge between indices [0,480]
    face_top_ind = [i for i in ind_bound if i < 480] # get indices of forehead on mask
    face_top = face[face_top_ind] # then obtian their coordinates, middle is no. 13
    
    neighbour_top = [] # index of nearneibour of forehead on top side of the mask
    neigh = []
    now = time.time()
    #get_mesh()
    
    for i in fore_ind:        
        temp = head[i - face_count] # get one point on fore head (head)
        # get distance from each fore head vertex to mask boundary
        dist = np.square(face_top[:,0] - temp[0]) + np.square(face_top[:,1] - temp[1]) + np.square(face_top[:,2] - temp[2])
        neighbour_top.append(face_top_ind[np.argmin(dist)]) # store index

    neigh = face[neighbour_top] # obtain coordinate    
    
    fore = head[fore_ind - face_count]
    #top_y = np.mean(neigh[:,1],axis = 0) - np.mean(fore[:,1],axis = 0) 
    top_z = np.mean(neigh[:,2],axis = 0) - np.mean(fore[:,2],axis = 0)

    face[:,2] -= top_z # align faace using forehead (working)
    update_mesh(face, head)
    #get_mesh()
    #! alignment done
    

    neigh = face[neighbour_top]
    top_y = neigh[12,1] - head[fore_ind[12] - face_count,1]
    #top_z = neigh[12,2] - head[fore_ind[12] - face_count,2]
    
    bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(fore_ind)  
    bpy.ops.transform.translate(value=(0, 0,top_y), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')
    #! jaw part
    
    # jaw_ind = np.loadtxt(os.path.join(DIR_KPTS, 'jaw.txt').astype(np.int32)
    #bpy.ops.mesh.select_all(action='DESELECT')
    sel_vert(jaw_ind)
    jaw = head[jaw_ind - face_count]
    temp2 = face[kpt_ind[8]]

    bottom_y = temp2[1] - np.mean(jaw[:,1],axis = 0) 
    bottom_z = temp2[2] - np.mean(jaw[:,2],axis = 0) 
    bpy.ops.transform.translate(value=(0,-bottom_z,-bottom_y), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='CONNECTED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.ops.mesh.select_all(action='DESELECT')
    

def get_kpts():
    #! load the vertex index correspond to facial landmarks
    # procedure refer to get_kpt_ind.py
    global kpt_ind, left_ind, fore_ind, jaw_ind, ind_bound, neck_ind
	DIR_KPTS = '.\\Data\\geometry'
    ind_bound = np.loadtxt(os.path.join(DIR_KPTS, 'bound.txt')).astype(np.int32) 
    kpt_ind = np.loadtxt(os.path.join(DIR_KPTS, 'kpt_ind.txt')).astype(np.int32) # ntri x 3
    left_ind = np.loadtxt(os.path.join(DIR_KPTS, 'ear.txt').astype(np.int32)
    fore_ind = np.loadtxt(os.path.join('fore_ind.txt')).astype(np.int32)
    jaw_ind = np.loadtxt(os.path.join(DIR_KPTS, 'jaw.txt')).astype(np.int32)
    neck_ind = np.loadtxt(os.path.join(DIR_KPTS, 'neck.txt')).astype(np.int32)


def get_scale():

    p1 = head[52447 - face_count]
    p2 = head[44683 - face_count]
    dis1 = abs(p1[0] - p2[0])
    #print(dis1)

    p21 = face[28003]
    p22 = face[27792]
    dis2 = abs(p21[0] - p22[0])
    #print(dis2)
    
    scal = dis2/dis1
    return scal
    
def get_pos():
    x = 0
    y = 0
    z = 0
    
    left  = head[left_ind - face_count]
    left_kpt = face[kpt_ind[14:17]]

    y = head[51673 - face_count,1] - face[kpt_ind[8],1]
    z = np.mean(left[:,2],axis = 0) - np.mean(left_kpt[:,2],axis=0)

    return (x,y,z)

def get_mesh():
    global ob, mesh, face, head
    ob   = bpy.data.objects['Object']
    bpy.context.scene.objects.active = ob #! required to select correct context
    #bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.object.mode_set(mode = 'EDIT')     
    mesh = bmesh.from_edit_mesh(ob.data) 
    mesh.verts.ensure_lookup_table()
    bpy.ops.mesh.select_all(action='DESELECT')

    face = []
    for i in range(face_count):
        face.append(mesh.verts[i].co)
        

    face = np.array(face) # coordinates of face
    head = []
    for i in range(face_count,59393):
        head.append(mesh.verts[i].co)
    head = np.array(head) # coordinates of face
    

def update_mesh(face_new, head_new):
    for i in range(face_count):
        mesh.verts[i].co = face_new[i,:]

    for ii in range(face_count,59393):
        mesh.verts[i].co = face_new[i,:]   

def import_face():
       
    import_obj('C:\\Users\\KTL\\Desktop\\FYP-code\\Data\\mask\\0.obj', 'Object')
    bpy.context.scene.objects.active = None

def import_head():
    
    file_loc = 'C:\\Users\\KTL\\Desktop\\FYP-code\\split\\head_dissolve.obj' #TODO change to more recent head model

    import_obj(file_loc,'head')
    select('head')
    bpy.context.scene.objects.active.scale = (1.165, 1.165, 1.165)
    bpy.context.scene.objects.active = None
    
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
    #bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.join(ctx)


def sel_vert(ind):
    ob = bpy.data.objects['Object']
    bpy.context.scene.objects.active = ob # select desired object first
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    mesh.verts.ensure_lookup_table()
    
    for i in ind:   
        mesh.verts[i].select = True

    #bpy.context.scene.objects.active = ob # refresh the scene

def Init():
    global face
    start = time.time()
    found = 'Object' in bpy.data.objects
    get_kpts()
    if not(found):
        
        import_head()
        import_face() # import face model
        now = time.time()
        print("import done: ",now-start)
        face_join()

    get_mesh()

    if not(found):
        now = time.time()
        print("merge done: ",now-start)
        #get_mesh()
        scal = get_scale()
        #print(scal)
        face = face/scal    
        #update_mesh(face, head)        
        #get_mesh()

        transx, transy, transz = get_pos()

        face[:,0] += 0
        face[:,1] += transy
        face[:,2] += transz
        now = time.time()
        print("rescale done: ",now-start)
        update_mesh(face, head)
        now = time.time()
        print("update done: ",now-start)

        #hide()
        edge_fit()
        now = time.time()
        
        
        bpy.ops.mesh.select_all(action='DESELECT')
        sel_vert(neck_ind)
        #select('Object')
        
        bpy.ops.mesh.delete(type='VERT')

        print("ALL done: ",now-start)
        bpy.ops.mesh.separate(type='MATERIAL')
        now = time.time()
        print("split done: ",now-start)
        ob1 = bpy.data.objects['Object.001']
        ob1.name = 'face'

  
face_count = 43866 # the number of vertex in face mask
Init()
