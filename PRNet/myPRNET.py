import numpy as np
import os
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
# import ast
from .api import PRN
# from .utils.estimate_pose import estimate_pose
from .utils.rotate_vertices import frontalize
from .utils.render_app import get_visibility, get_uv_mask
from .utils.write import write_obj_with_colors, write_obj_with_texture
import cv2


# def main(args):
#     start_time = time()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
#     save_folder = args.outputDir
#     if not os.path.exists(save_folder):
#         os.mkdir(save_folder)
#
#     image_path = args.img_path
#
#     prn = PRN(is_mtcnn = args.isMTCNN)
#     name = image_path.strip().split('/')[-1][:-4]
#     # read image
#     image = imread(image_path)
#     [h, w, c] = image.shape
#     if c>3:
#         image = image[:,:,:3]
#     # the core: regress position map
#     if args.isMTCNN:
#         max_size = max(image.shape[0], image.shape[1])
#         if max_size> 1000:
#             image = rescale(image, 1000./max_size)
#             image = (image*255).astype(np.uint8)
#         pos = prn.process(image) # use dlib to detect face
#     else:
#         if image.shape[0] == image.shape[1]:
#             image = resize(image, (256,256))
#             pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
#         else:
#             box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
#             pos = prn.process(image, box)
#
#     image = image/255.
#     if pos is None:
#         return
#     # 3D vertices
#     vertices = prn.get_vertices(pos)
#     if args.isFront:
#         save_vertices = frontalize(vertices)
#     else:
#         save_vertices = vertices.copy()
#     save_vertices[:,1] = h - 1 - save_vertices[:,1]
#
#     # corresponding colors
#     colors = prn.get_colors(image, vertices)
#     if args.isTexture:
#         if args.texture_size != 256:
#             pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
#         else:
#             pos_interpolated = pos.copy()
#         texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
#         if args.isMask:
#             vertices_vis = get_visibility(vertices, prn.triangles, h, w)
#             uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
#             uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
#             texture = texture*uv_mask[:,:,np.newaxis]
#         write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
#     else:
#         write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
#     if args.isKpt:
#         # get landmarks
#         kpt = prn.get_landmarks(pos)
#         np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
#     end_time = time()
#     print("Time Elasped {}".format(end_time-start_time))

def genPRMask(image_path, save_folder='temp', isMTCNN=True, isFront=True, isTexture=True, isMask=True, isKpt=False,
              isCrop=True, texture_size=256):
    start_time = time()
    # os.environ['CUDA_VISIBLE_DEVICES'] = 0
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    prn = PRN(is_mtcnn=isMTCNN)
    name = image_path.strip().split('\\')[-1][:-4]
    # read image
    try:
        image = imread(image_path)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:, :, :3]
        # the core: regress position map
        if isMTCNN:
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000. / max_size)
                image = (image * 255).astype(np.uint8)
            pos = prn.process(image)  # detect face
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256, 256))
                pos = prn.net_forward(image / 255.)  # input image has been cropped to 256x256
            else:
                box = np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # cropped with bounding box
                pos = prn.process(image, box)

        image = image / 255.
        if pos is None:
            return
        # 3D vertices
        vertices = prn.get_vertices(pos)
        if isFront:
            save_vertices = frontalize(vertices)
        else:
            save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]
        colors = prn.get_colors(image, vertices)
        if isTexture:
            if texture_size != 256:
                pos_interpolated = resize(pos, (texture_size, texture_size), preserve_range=True)
            else:
                pos_interpolated = pos.copy()
            texture = cv2.remap(image, pos_interpolated[:, :, :2].astype(np.float32), None,
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
            if isMask:
                vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                uv_mask = resize(uv_mask, (texture_size, texture_size), preserve_range=True)
                texture = texture * uv_mask[:, :, np.newaxis]
            write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture,
                                   prn.uv_coords / prn.resolution_op)  # save 3d face with texture(can open with meshlab)
        else:
            write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles,
                                  colors)  # save 3d face(can open with meshlab)
        if isKpt:
            # get landmarks
            kpt = prn.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
        end_time = time()
    except Exception as e:
        print(e)
        raise e
    return {"obj": os.path.join(save_folder, name + '.obj'), "mtl": os.path.join(save_folder, name + '.mtl'),
            "texture": os.path.join(save_folder, name + '_texture.png')}

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
#     parser.add_argument('img_path', type=str, help='path to the input image')
#     parser.add_argument('-o', '--outputDir', default='temp', type=str, help='path to the output directory, where results(obj,txt files) will be stored.')
#     parser.add_argument('--gpu', default='0', type=str,
#                         help='set gpu id, -1 for CPU')
#     parser.add_argument('--isMTCNN', default=True, type=ast.literal_eval,
#                         help='whether to use MTCNN for detecting face, default is True, if False, the input image should be cropped in advance')
#     parser.add_argument('--is3d', default=True, type=ast.literal_eval,
#                         help='whether to output 3D face(.obj). default save colors.')
#     parser.add_argument('--isKpt', default=True, type=ast.literal_eval,
#                         help='whether to output key points(.txt)')
#     parser.add_argument('--isFront', default=True, type=ast.literal_eval,
#                         help='whether to frontalize vertices(mesh)')
#     parser.add_argument('--isTexture', default=True, type=ast.literal_eval,
#                         help='whether to save texture in obj file')
#     parser.add_argument('--isMask', default=True, type=ast.literal_eval,
#                         help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
#     # update in 2017/7/19
#     parser.add_argument('--texture_size', default=256, type=int,
#                         help='size of texture map, default is 256. need isTexture is True')
#     main(parser.parse_args())
