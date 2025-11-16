import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from utils.transforms import make_transforms_clouds
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
from copy import deepcopy
from transformers import CLIPImageProcessor

import imageio
import cv2
import random
import json

from utils.utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from model.llava import conversation as conversation_lib

from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)

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
        pc,
        coords,
        feats,
        # sem_labels,
        # inst_labels,
        inverse_indexes,
        scene_name,
        gt_data,
    ) = list(zip(*batch))

    len_batch = []
    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        len_batch.append(N)


    coords = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats = torch.cat(feats, dim=0)
    # imgs = torch.cat(imgs, dim=0)

    # sem_labels = torch.cat(sem_labels, 0).long()
    # inst_labels = torch.cat(inst_labels, 0).long()
    # print(f"sem_labels shape: {sem_labels.shape}")
    # print(f"inst_labels shape: {inst_labels.shape}")
    # 合并 gt_data 中的内容
    masks_points = []
    object_id_tensors = []
    descriptions = []
    image_filenames = []
    offset_list = [0]
    for i, data in enumerate(gt_data):
        num_instructions = data['masks_points'].shape[0]
        masks_points.append(data['masks_points'])
        object_id_tensors.extend(data['object_id_tensors'])
        descriptions.extend(data['descriptions'])
        image_filenames.extend(data['image_filenames']) 
        # object_names.extend(data['object_names'])
        # 更新 offset_list
        offset_list.extend([offset_list[-1] + num_instructions])

    masks_points = torch.cat(masks_points, dim=0)
    offset_list = torch.LongTensor(offset_list)


    return {
        "pc": pc,  # point cloud
        "sinput_C": coords,  # discrete coordinates (ME)
        "sinput_F": feats,  # point features (N, 3)
        # "input_I": imgs,
        "len_batch": len_batch,
        # "sem_labels": sem_labels, # labels for each point
        # "inst_labels": inst_labels,  # labels for each point
        "inverse_indexes": inverse_indexes, 
        "lidar_name": scene_name,
        # gt_data
        "masks_points": masks_points, # shape(total_instructions, total_points)
        "object_id_tensors": object_id_tensors, # list，长度为total_instructions
        "descriptions": descriptions,  # 
        "image_filenames": image_filenames,  # 添加图像文件名
        "offset_list": offset_list, #用于指示每个场景中指令的起始和结束索引
    }

class scannetpp_Dataset(Dataset):
    def __init__(self, phase, config, transforms = None):

        self.scannet_root_dir = config['dataRoot_scannetpp']
        if phase == 'train':
            self.scannet_file_list = self.read_files(config['train_file'])

        else:
            self.scannet_file_list = self.read_files(config['val_file'])
        # Load descriptions from train.json or val.json
        instruction_file = config['train_description'] if phase == 'train' else config['val_description']
        with open(instruction_file, 'r') as f:
            instruction3D = json.load(f)
            
        # {
        #     "scene_id": "f8f12e4e6b",
        #     "object_id": [
        #         116
        #     ],
        #     "object_name": "bookshelf",
        #     "description": "This item is used to store books. It is the one near the door."
        # },

        self.instruction3D = {}
        for item in instruction3D:
            scene_name = item["scene_id"]
            if scene_name not in self.instruction3D:
                self.instruction3D[scene_name] = []  # 初始化一个新的列表
            self.instruction3D[scene_name].append({
                "object_id": item["object_id"],
                "object_name": item["object_name"],
                "image": item["image"],
                "description": item["description"]
            })
        print("instruction3D scene number: ", len(self.scannet_file_list))

        self.points_path = config['points_path']

        # self.image_transforms = image_transforms
        # self.mixed_transforms = mixed_transforms

        self.voxel_size = config['voxel_size']
        self.phase = phase
        self.config = config
        self.imageDim = (1752, 1168)
        self.transforms = transforms
        self.maxImages = 8

        # self.tokenizer = tokenizer

        # self.short_question_list = SHORT_QUESTION_LIST
        # self.long_question_list = LONG_QUESTION_LIST
        # self.answer_list = ANSWER_LIST
        # self.ignore_label = 255

        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

        # self.scene_buffer = {}
        


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

    def load_point_cloud_data(self, data_path):
        data = torch.load(data_path)
        coords = torch.tensor(data['sampled_coords'], dtype=torch.float32)
        feats = torch.tensor(data['sampled_colors'], dtype=torch.float32)
        sem_labels = torch.tensor(data['sampled_labels'], dtype=torch.int8)
        inst_labels = torch.tensor(data['sampled_instance_labels'], dtype=torch.int8)
        anno_id = torch.tensor(data['sampled_instance_anno_id'], dtype=torch.int8)
        # print(f"Shapes: coords={coords.shape}, feats={feats.shape}, sem_labels={sem_labels.shape}, inst_labels={inst_labels.shape}, anno_id={anno_id.shape}")
        
        return coords, feats, sem_labels, inst_labels, anno_id

    def get_gt_mask(self, scene_name, inst_labels):

        instructions = self.instruction3D[scene_name]

        # instructions = random.sample(instructions, min(args.num_classes_per_sample, len(instructions)))
        # instructions = instructions[:2]
        num_instructions = len(instructions)  #
        instructions = instructions[:num_instructions]
        # print(len(instructions), instructions)
        # instructions[0]["description"] = "Based on the picture, on which part of the body is it most likely to have a permanent decorative design on the skin?"
        # instructions[1]["description"] = "Based on the picture, which part of the body of the person is likely to have a permanent decorative design on the skin?"
        # instructions[2]["description"] = "Looking at the person in the picture, what part of her body is likely to have a decorative design on the skin permanently?"
        # instructions[0]["description"] = "On the kitchen counter, what appliance is used for toasting bread?"
        # instructions[1]["description"] = "What object attached to the sink is used to control the flow of water?"

        object_id_tensors = []
        for instruction in instructions:
            object_id_tensors.append(torch.tensor(instruction["object_id"]))

        sampled_sents = []
        sampled_mask_point = []
        sampled_object_names = []
        sampled_images = []  #

        for instruction in instructions:
            # object_id, object_name, description
            object_id = instruction["object_id"]
            object_name = instruction["object_name"]
            mask, sent = None, instruction["description"]
            image_filename = instruction["image"]
            # image_clip = None
            # 组合 description 和 object_name
            # sent = f"{description} The answer is {object_name}. Please output the segmentation mask."

            mask_point = np.zeros(inst_labels.shape[0], dtype=np.int64)
            for id in object_id:
                mask_point[inst_labels == id] = 1

            # sampled_masks.append(mask)
            sampled_sents.append(sent)
            # images_clip.append(image_clip)
            sampled_mask_point.append(mask_point.copy())
            sampled_images.append(image_filename) 
            # sampled_object_names.append(object_name)
            # print(mask_point.sum())

        # print("sampled_masks: ", len(sampled_masks), sampled_masks[0].max(), sampled_masks[0].min())
        # print("sampled_sents: ", len(sampled_sents), sampled_sents)
        # print("sampled_mask_point: ", len(sampled_mask_point), sampled_mask_point[0].shape)
        # print("sampled_object_names:", len(sampled_object_names), sampled_object_names)

        masks_points = torch.from_numpy(np.stack(sampled_mask_point, axis=0))

        # packeging
        # images_clip_list = images_clip
        # conversation_list = conversations
        offset_list = [0]
        for i in range(num_instructions):
            offset_list.append(offset_list[-1] + 1)  # 每个指令的掩码数量为 1
        offset_list = torch.LongTensor(offset_list)
        # 返回值中包含 offset_list        



        # print("offset: ", offset_list.shape)


        # LISA_input_dict = {
        #     # "image_paths": image_path_list,
        #     # "images": torch.stack(images_list, dim=0),
        #     # "images_clip": torch.stack(images_clip_list, dim=0),
        #     # "input_ids": input_ids,
        #     # "labels": targets,
        #     # "attention_masks": attention_masks,
        #     # "masks_images": masks_images,
        #     # "label_list": label_list,
        #     # "resize_list": resize_list,
        #     "offset": torch.LongTensor(offset_list),
        #     "masks_points": masks_points,
        #     "object_id_tensors": object_id_tensors,
        #     "object_names": sampled_object_names,
        #     # "questions_list": questions_list,
        #     # "sampled_classes_list": sampled_classes_list,
        #     # "inference": inferences[0],
        #     # "conversation_list": conversation_list,
        #     # "sampled_cameras_info": sampled_cameras_info
        # }

        return {
            "offset": offset_list,
            "masks_points": masks_points,
            "object_id_tensors": object_id_tensors,
            # "object_names": sampled_object_names,
            "descriptions": sampled_sents,
            "image_filenames": sampled_images
        }

    def __getitem__(self, idx):
        scene_name = self.scannet_file_list[idx]
        data_path = os.path.join(self.scannet_root_dir, scene_name)
        points_path = os.path.join(self.scannet_root_dir, 'prepare_sem_data', f"{self.scannet_file_list[idx]}.pth")

        coords, feats, sem_labels, inst_labels, anno_id = self.load_point_cloud_data(points_path)
        # print(sem_labels.shape, inst_labels.shape)

        sem_labels = sem_labels.to(torch.int16)
        inst_labels = inst_labels.to(torch.int64)
        anno_id = anno_id.to(torch.int16)

        sem_labels[sem_labels == -100] = -1
        sem_labels += 1

        gt_data = self.get_gt_mask(scene_name, inst_labels)
        # print(gt_data)

        pc = coords.clone()
        # coords, labels = data[:, :3], data[:, 9:]
        # sceneName = self.scannet_file_list[idx]

        # write_ply_rgb(coords, feats, "visual/visual_%s.ply" % sceneName)

        feats = feats / 127.5 - 1
        coords = (coords - coords.mean(0)) / self.voxel_size

        # print(feats)
        # feats = torch.ones(len(coords), 1)

        '''
        # print image-point correspondence
        img_pixel = tuple(pairing_image.T.long())
        img_RGB = img[img_pixel]
        print(coords[pairing_point].shape, "img_RGB ", img_RGB.shape)
        write_ply_rgb(coords[pairing_point], img_RGB*255, "visual/visual_%s_%s.ply" % (frameid, i))
        '''

        if self.transforms:
            coords = self.transforms(coords.float())
        # coords = torch.from_numpy(coords)
        # if self.cloud_transforms:
        #     coords = self.cloud_transforms(coords)

    
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        # pairing_points = inverse_indexes[pairing_points]

        feats = feats[indexes]
        sem_labels = sem_labels[indexes]
        inst_labels = inst_labels[indexes]
        # assert pairing_points.shape[0] == pairing_images.shape[0]

        packages = (pc, discrete_coords, feats, inverse_indexes, scene_name, gt_data)
        # packages = (pc, discrete_coords, feats, sem_labels, inst_labels, inverse_indexes, scene_name, gt_data)
        return packages


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = scannetpp_Dataset(phase=phase, transforms=transforms, config=config)

    collate_fn = scannetpp_collate_pair_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
