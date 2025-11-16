import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import MinkowskiEngine as ME
from torch.utils.data import Dataset
# import pc_utils
from plyfile import PlyData, PlyElement
import math
# from pc_utils import write_ply_rgb
import sys
sys.path.append("../..")

from random import randint
import uuid
from argparse import Namespace

import imageio
import cv2
import random
import json


def write_ply_rgb(points, colors, filename, text=True):
    """ input: Nx3, Nx3 write points and colors to filename as PLY format. """
    num_points = len(points)
    assert len(colors) == num_points

    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    colors = [(colors[i, 0], colors[i, 1], colors[i, 2]) for i in range(colors.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    color = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(num_points, vertex.dtype.descr + color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in color.dtype.names:
        vertex_all[prop] = color[prop]

    el = PlyElement.describe(vertex_all, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def scannetpp_collate_pair_fn(batch):

    (
        coords,
        feats,
        imgs,
        pairing_points,
        pairing_images,
        inverse_indexes,
        pseudo_data,
    ) = list(zip(*batch))

    offset_point = 0
    offset_image = 0

    for batch_id in range(len(coords)):
        #print(f"pairing_points[{batch_id}]: {pairing_points[batch_id]}, type: {type(pairing_points[batch_id])}")
        #print(f"offset_point: {offset_point}, type: {type(offset_point)}")

        pairing_points[batch_id][:] += offset_point
        offset_point += coords[batch_id].shape[0]

        pairing_images[batch_id][:, 0] += offset_image
        offset_image += imgs[batch_id].shape[0]

    len_batch = []
    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        len_batch.append(N)


    coords = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats = torch.cat(feats, dim=0)
    pairing_points = torch.cat(pairing_points, dim=0)
    pairing_images = torch.cat(pairing_images, dim=0)
    imgs = torch.cat(imgs, dim=0)

    # 合并 pseudo_data 中的内容
    # masks_points = []
    # object_id_tensors = []
    descriptions = []
    image_filenames = []
    offset_list = [0]
    for i, data in enumerate(pseudo_data):
        # num_instructions = data['descriptions']
        # masks_points.append(data['masks_points'])
        # object_id_tensors.extend(data['object_id_tensors'])
        descriptions.extend(data['descriptions'])
        image_filenames.extend(data['image_filenames']) 
        # object_names.extend(data['object_names'])
        # 更新 offset_list
        # offset_list.extend([offset_list[-1] + num_instructions])

    # masks_points = torch.cat(masks_points, dim=0)
    # offset_list = torch.LongTensor(offset_list)


    return {
        "sinput_C": coords,  # discrete coordinates (ME)
        "sinput_F": feats,  # point features (N, 3)
        "input_I": imgs, # 每个description对应的图像
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        # "len_batch": len_batch,
        "inverse_indexes": inverse_indexes, 
        "descriptions": descriptions,  # 
        # "image_filenames": image_filenames,  # 图像文件名
        # "offset_list": offset_list, #用于指示每个场景中指令的起始和结束索引
    }

class scannetpp_Dataset(Dataset):
    def __init__(self, phase, config, shuffle = True, image_transforms = None, cloud_transforms = None, mixed_transforms = None):

        self.scannet_root_dir = config['dataRoot_scannetpp']
        if phase == 'train':
            self.scannet_file_list = self.read_files(config['train_file'])
            # self.scannet_file_list = self.read_files(config['val_file'])
        else:
            self.scannet_file_list = self.read_files(config['val_file'])
        
        instruction_file = config['train_description'] if phase == 'train' else config['val_description']
        with open(instruction_file, 'r') as f:
            instruction3D = json.load(f)

        # {
        #     "scene_id": "c4c04e6d6c",
        #     "object_id": [
        #         76,
        #         77,
        #         9,
        #         8
        #     ],
        #     "object_name": "blackboard",
        #     "image": "DSC03393.JPG",
        #     "description": "When ideas need to be shared or a lesson taught in an educational setting, which large, flat surface serves as the writing board?"
        # },            
        self.instruction3D = {}
        for item in instruction3D:
            scene_name = item["scene_id"]
            if scene_name not in self.instruction3D:
                self.instruction3D[scene_name] = []  # 初始化一个新的列表
            self.instruction3D[scene_name].append({
                "object_id": item["object_id"],
                "object_name": item["object_name"],
                "images": item["images"],
                "description": item["description"]
            })
        # print("instruction3D scene number: ", len(self.scannet_file_list))

        self.image_transforms = image_transforms
        self.mixed_transforms = mixed_transforms

        self.voxel_size = config['voxel_size']
        self.phase = phase
        self.config = config
        self.imageDim = (1752, 1168)
        self.cloud_transforms = cloud_transforms

    def read_files(self, scene_file):
        with open(scene_file, 'r') as file: scenes = file.readlines()
        scenes = [scene.split('\n')[0] for scene in scenes]
        return scenes

    def __len__(self):
        return len(self.scannet_file_list)

    def read_intrinsic_file(self, fname):
        with open(fname, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # ignore "#" 
            if line.startswith("#") or not line.strip():
                continue

            tokens = line.split()
            if len(tokens) >= 12:
            
                camera_id = int(tokens[0])
                model = tokens[1]
                width = int(tokens[2])
                height = int(tokens[3])
                params = list(map(float, tokens[4:]))

                f_x = params[0] 
                f_y = params[1]  
                c_x = params[2] 
                c_y = params[3]  

                intrinsic = np.array([
                    [f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0, 0, 1]
                ])

                return intrinsic

        raise ValueError("No valid camera parameters found in file.")


    def read_txt(self, path):
        # Read txt file into lines.
        with open(path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def read_images_file(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()

        images = {}
        idx = 0

        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith('#') or line == '':
                idx += 1
                continue

            tokens = line.split()
            image_id = int(tokens[0])
            qw, qx, qy, qz = map(float, tokens[1:5])  # rotation
            tx, ty, tz = map(float, tokens[5:8])      # trans
            camera_id = int(tokens[8])
            image_name = tokens[9]

            # rotation matrix
            rotation = R.from_quat([qx, qy, qz, qw])  # [qx, qy, qz, qw]
            rotation_matrix = rotation.as_matrix()    # 3x3 

            # world_to_camera
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = [tx, ty, tz]

            idx += 1
            points2d_line = lines[idx].strip()
            points2d_tokens = points2d_line.split()
            points2d = []
            for i in range(0, len(points2d_tokens), 3):
                x = float(points2d_tokens[i])
                y = float(points2d_tokens[i + 1])
                point3d_id = int(points2d_tokens[i + 2])
                points2d.append((x, y, point3d_id))

            images[image_name] = {
                'pose': pose,  #  (world_to_camera)
                'camera_id': camera_id,
                'image_id': image_id,
                'points2d': points2d
            }

            idx += 1

        return images

    def computeLinking(self, world_to_camera, coords, depth, link_proj_threshold, intrinsic, imageDim):
        """
        :param camera_to_world: 4 x 4, extrinsics
        :param coords: N x 3 format
        :param depth: H x W format
        :link_proj_threshold: Threshold for determining occlusions
        :intrinsic: 4 x 4
        :return: linking, N x 3 format, (H,W,mask)
        """

        # print("imageDim ", imageDim)
        
        link = np.zeros((3, coords.shape[0]), dtype=float)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T #4 x N
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        p = np.matmul(world_to_camera, coordsNew) # 4 x N

        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]

        pi = p
        # inside_mask = (pi[0] >= 0) * (pi[1] >= 0) * (pi[0] <= imageDim[1] - 1) * (pi[1] <= imageDim[0]-1)
        inside_mask = (pi[0] >= 0) & (pi[1] >= 0) & (pi[0] <= imageDim[1] - 1) & (pi[1] <= imageDim[0] - 1)

        occlusion_mask = np.abs(depth[np.round(pi[1][inside_mask]).astype(int), np.round(pi[0][inside_mask]).astype(int)] - p[2][inside_mask]) <= link_proj_threshold

        # inside_mask[inside_mask == True] = occlusion_mask
        inside_mask[inside_mask] &= occlusion_mask

        link[0][inside_mask] = pi[1][inside_mask] 
        link[1][inside_mask] = pi[0][inside_mask] 
        link[2][inside_mask] = 1 

        return link.T

    def prepare_output_and_logger(self, args):
        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str = os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])

        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
            
    def load_point_cloud_data(self, data_path):
        data = torch.load(data_path)
        coords = torch.tensor(data['sampled_coords'], dtype=torch.float32)
        feats = torch.tensor(data['sampled_colors'], dtype=torch.float32)
        # sem_labels = torch.tensor(data['sampled_labels'], dtype=torch.int8)
        # inst_labels = torch.tensor(data['sampled_instance_labels'], dtype=torch.int8)
        # anno_id = torch.tensor(data['sampled_instance_anno_id'], dtype=torch.int8)
        # print(f"Shapes: coords={coords.shape}, feats={feats.shape}, sem_labels={sem_labels.shape}, inst_labels={inst_labels.shape}, anno_id={anno_id.shape}")
        
        return coords, feats
    
    def get_pseudo_data(self, scene_name, num_classes_per_sample):
        
        instructions = self.instruction3D[scene_name]
        instructions = random.sample(instructions, min(num_classes_per_sample, len(instructions)))
        # instructions = instructions[:2]
        num_instructions = len(instructions)  
        # instructions = instructions[:num_instructions]
        # print(len(instructions), instructions)
        object_id_tensors = []
        for instruction in instructions:
            object_id_tensors.append(torch.tensor(instruction["object_id"]))

        sampled_sents = []
        sampled_mask_point = []
        sampled_object_names = []
        sampled_images = []  

        for instruction in instructions:
            # object_id, object_name, description
            object_id = instruction["object_id"]
            object_name = instruction["object_name"]
            mask, sent = None, instruction["description"]
            image_filename = instruction["images"]
           
            # mask_point = np.zeros(inst_labels.shape[0], dtype=np.int64)
            # for id in object_id:
            #     mask_point[inst_labels == id] = 1

            # sampled_masks.append(mask)
            sampled_sents.append(sent)
            # sampled_mask_point.append(mask_point.copy())
            sampled_images.append(image_filename) 
            # sampled_object_names.append(object_name)
            # print(mask_point.sum())

        offset_list = [0]
        for i in range(num_instructions):
            offset_list.append(offset_list[-1] + 1)  # 每个指令的掩码数量为 1
        offset_list = torch.LongTensor(offset_list)
        # 返回值中包含 offset_list   
        # 
        return {
            "offset": offset_list,
            # "masks_points": masks_points,
            # "object_id_tensors": object_id_tensors,
            # "object_names": sampled_object_names,
            "descriptions": sampled_sents,
            "image_filenames": sampled_images
        }     
 
    def __getitem__(self, idx):
        scene_name = self.scannet_file_list[idx]
        data_path = os.path.join(self.scannet_root_dir, scene_name)
        points_path = os.path.join(self.scannet_root_dir, 'prepare_sem_data', f"{self.scannet_file_list[idx]}.pth")

        coords, feats = self.load_point_cloud_data(points_path)
        # print(sem_labels.shape, inst_labels.shape)
        feats = feats / 127.5 - 1

        pairing_points = []
        pairing_images = []
        imgs = []

        scene_name = self.scannet_file_list[idx]
        dslr_path = os.path.join(self.scannet_root_dir, 'data', scene_name, 'dslr')
        image_path = os.path.join(dslr_path, 'undistorted_images')
        depth_path = os.path.join(dslr_path, 'undistorted_depths')
        sparse_path = os.path.join(self.scannet_root_dir, 'data', scene_name, 'sparse/0')
        pose_file = os.path.join(sparse_path, 'images.txt')
        camera_file = os.path.join(sparse_path, 'cameras.txt')

        pseudo_data = self.get_pseudo_data(scene_name, 4)

        frame_names = pseudo_data["image_filenames"]

        for i, frameid in enumerate(frame_names):
            img_file = os.path.join(image_path, f"{frameid}.JPG")
            img = cv2.imread(img_file) # Read original image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.image_transforms:
                # img, applied_transforms = self.image_transforms(img)
                img = self.image_transforms(img)
            
            img = cv2.resize(img / 255, self.imageDim) # img shape: (1168, 1752, 3)
            # print(f"img shape: {img.shape}")
            depth = imageio.imread(os.path.join(depth_path, f"{frameid}.png")) / 1000.0
            images_info = self.read_images_file(pose_file)
            pose = images_info[f"{frameid}.JPG"]['pose']
            intrinsic = self.read_intrinsic_file(camera_file)

            link = self.computeLinking(pose, coords, depth, 0.05, intrinsic, depth.shape)

            pairing_point = torch.from_numpy(np.argwhere(link[:, 2] == 1)).squeeze()
            # print(f"pairing_point shape: {pairing_point.shape}")
            pairing_points.append(pairing_point)

            link = torch.from_numpy(link).int()
            imgs.append(torch.from_numpy(img.transpose((2, 0, 1)))) # (W, H, C) --> (C, W, H)
            #imgs.append(torch.from_numpy(img))

            pairing_image = link[pairing_point, :2]
            # print(f"pairing_image shape: {pairing_image.shape}")
            pairing_images.append(torch.cat((torch.ones(pairing_point.shape[0], 1) * i,
                                            pairing_image), dim=1))
        imgs = torch.stack(imgs)                 
        pairing_points = torch.cat(pairing_points, dim=0).numpy()
        pairing_images = torch.cat(pairing_images, dim=0).numpy()
        #print(f"pairing_points shape: {pairing_points.shape}")
        #print(f"pairing_images shape: {pairing_images.shape}")                   

        if self.cloud_transforms:
            coords = self.cloud_transforms(coords.float())

        if self.mixed_transforms:
            (
                coords_b,
                feats_b,
                imgs_b,
                pairing_points_b,
                pairing_images_b,
            ) = self.mixed_transforms(
                coords, feats, imgs, pairing_points, pairing_images
            )

        coords, feats, imgs, pairing_points, pairing_images = coords_b, feats_b, imgs_b.transpose(1, 2).transpose(2, 3), torch.from_numpy(pairing_points_b),\
            torch.from_numpy(pairing_images_b)

        coords = (coords - coords.mean(0)) / self.voxel_size
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        feats = feats[indexes]
        assert pairing_points.shape[0] == pairing_images.shape[0]

        packages = (discrete_coords, feats, imgs, pairing_points, pairing_images, inverse_indexes, pseudo_data
        )
        return packages

