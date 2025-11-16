import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from torch.utils.data import Dataset
# import pc_utils
from utils.transforms import make_transforms_clouds
from plyfile import PlyData, PlyElement
import math
# from pc_utils import write_ply_rgb
import sys
sys.path.append("..")
# from MinkowskiEngine.utils import sparse_quantize

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


def scannet_collate_pair_fn(batch):
    (
        pc,
        coords,
        feats,
        gt_data,
        inverse_indexes,
        scan_names,
    ) = list(zip(*batch))

    len_batch = [coords[i].shape[0] for i in range(len(coords))]

    coords = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats = torch.cat(feats, dim=0)

    masks_points = torch.cat([data['masks_points'] for data in gt_data], dim=0)
    object_id_tensors = [tensor for data in gt_data for tensor in data['object_id_tensors']]
    descriptions = [desc for data in gt_data for desc in data['descriptions']]

    # Compute offsets for masks
    offset_list = torch.cumsum(torch.tensor([data['masks_points'].shape[0] for data in gt_data]), dim=0).tolist()
    offset_list = [0] + offset_list

    # Compute num_description per scene
    num_description = [len(data['descriptions']) for data in gt_data]

    return {
        "pc": pc,
        "sinput_C": coords,
        "sinput_F": feats,
        "len_batch": len_batch,
        "inverse_indexes": inverse_indexes,
        "descriptions": descriptions,
        "lidar_name": scan_names,
        "masks_points": masks_points,
        # "object_id_tensors": object_id_tensors,
        "offset_list": offset_list,
        "num_description": num_description,  # Number of descriptions per scene
    }


class scannet_Dataset(Dataset):
    def __init__(self, phase, config, transforms=None):

        self.scannet_root_dir = config['dataRoot_scannet']
        if phase == 'train':
            self.scannet_file_list = self.read_files(config['train_file'])

            skip_ratio = config["dataset_skip_step"]
            print("before: ", len(self.scannet_file_list))
            self.scannet_file_list = sorted(self.scannet_file_list)[::skip_ratio]
            print("after: ", len(self.scannet_file_list))

        else:
            self.scannet_file_list = self.read_files(config['val_file'])

        instruction_dir = config['dataRoot_description']  # Directory containing description files
        all_instructions = {}

        # Iterate over all JSON files in the directory
        for json_file in os.listdir(instruction_dir):
            if json_file.endswith(".json"):  # Ensure only .json files are processed
                file_path = os.path.join(instruction_dir, json_file)
                with open(file_path, 'r') as f:
                    instruction_data = json.load(f)
                    for item in instruction_data:
                        scene_name = item["scene_id"]
                        if scene_name not in all_instructions:
                            all_instructions[scene_name] = []  # Initialize a new list
                        all_instructions[scene_name].append({
                            "object_ids": item["object_ids"],
                            "description": item["description"]
                        })

        # Filter instructions to include only those in scannet_file_list
        self.instruction3D = {
            scene_name: all_instructions[scene_name]
            for scene_name in self.scannet_file_list if scene_name in all_instructions
        }

        # instruction_file = config['dataRoot_description'] 
        # if phase == 'train':
        #     self.scannet_file_list = self.read_files(config['train_file'])
            
        # else:
        #     self.scannet_file_list = self.read_files(config['val_file'])

        # with open(instruction_file, 'r') as f:
        #     instruction3D = json.load(f)

        # self.instruction3D = {}
        # for item in instruction3D:
        #     scene_name = item["scene_id"]
        #     if scene_name not in self.instruction3D:
        #         self.instruction3D[scene_name] = []  # 初始化一个新的列表
        #     self.instruction3D[scene_name].append({
        #         "object_ids": item["object_ids"],
        #         # "object_name": item["object_name"],
        #         # "image": item["image"],
        #         "description": item["description"]
        #     })
        # print("instruction3D scene number: ", len(self.scannet_file_list))

        # self.points_path = config['points_path']

        self.voxel_size = config['voxel_size']
        self.phase = phase
        self.config = config
        self.imageDim = (640, 480)
        self.transforms = transforms
        self.maxImages = 8

    def read_files(self, file):
        f = open(file)
        lines = f.readlines()
        name_list = [line.split('.')[0] for line in lines]
        f.close()
        return name_list

    def __len__(self):
        return len(self.scannet_file_list)

    def read_pose_file(self, fname):
        posemat = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                              (x.split(" ") for x in open(fname).read().splitlines())])
        return posemat

    def read_intrinsic_file(self, fname):
        intrinsic = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                                (x.split(" ") for x in open(fname).read().splitlines())])
        return intrinsic

    def read_txt(self, path):
        # Read txt file into lines.
        with open(path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def computeLinking(self, camera_to_world, coords, depth, link_proj_threshold, intrinsic_color, intrinsic_depth, imageDim):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :intrinsic_depth: 4 x 4
        :intrinsic_color: 4 x 4, not used currently
        :return: linking, N x 3 format, (H,W,mask)
        """

        # print("imageDim ", imageDim)

        intrinsic = intrinsic_depth
        link = np.zeros((3, coords.shape[0]), dtype=float)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T #4 x N
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world) # 4 x 4
        p = np.matmul(world_to_camera, coordsNew) # 4 x N
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]

        pi = p
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) * (pi[0] <= imageDim[1] - 1) * (pi[1] <= imageDim[0]-1)

        occlusion_mask = np.abs(depth[np.round(pi[1][inside_mask]).astype(np.int), np.round(pi[0][inside_mask]).astype(np.int)] - p[2][inside_mask]) <= link_proj_threshold

        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1
        return link.T

    def get_gt_mask(self, scene_name, inst_labels):

        instructions = self.instruction3D[scene_name]

        num_instructions = len(instructions)  #
        instructions = instructions[:num_instructions]
     
        object_id_tensors = []
        for instruction in instructions:
            object_id_tensors.append(torch.tensor(instruction["object_ids"]))

        sampled_sents = []
        sampled_mask_point = []
        sampled_object_names = []
        sampled_images = []  #

        for instruction in instructions:
            # object_id, object_name, description
            object_ids = instruction["object_ids"]
            # object_name = instruction["object_name"]
            mask, sent = None, instruction["description"]
            # image_filename = instruction["image"]
            # image_clip = None
            # 组合 description 和 object_name
            # sent = f"{description} The answer is {object_name}. Please output the segmentation mask."

            mask_point = np.zeros(inst_labels.shape[0], dtype=np.int64)
            for id in object_ids:
                mask_point[inst_labels == id] = 1

            # sampled_masks.append(mask)
            sampled_sents.append(sent)
            # images_clip.append(image_clip)
            sampled_mask_point.append(mask_point.copy())
            # sampled_images.append(image_filename) 
            # sampled_object_names.append(object_name)
            # print(mask_point.sum())

        masks_points = torch.from_numpy(np.stack(sampled_mask_point, axis=0))

        # packeging
        # images_clip_list = images_clip
        # conversation_list = conversations
        offset_list = [0]
        for i in range(num_instructions):
            offset_list.append(offset_list[-1] + 1)  # 每个指令的掩码数量为 1
        offset_list = torch.LongTensor(offset_list)
        # 返回值中包含 offset_list        

        return {
            "offset": offset_list,
            "masks_points": masks_points,
            # "object_id_tensors": object_id_tensors,
            # "object_names": sampled_object_names,
            "descriptions": sampled_sents,
            # "image_filenames": sampled_images
        }
    
    def get_3D_mask(self, scene_name, instance_labels):
        """
        Generate the binary 3D ground truth mask for the given scene and instructions.

        Args:
            scene_name (str): Name of the scene.
            instance_labels (Tensor): Instance labels for the scene points.

        Returns:
            dict: Contains binary masks for points, descriptions, and offsets.
        """
        instructions = self.instruction3D[scene_name]
        sampled_mask_point = []
        sampled_sents = []
        object_id_tensors = []

        for instruction in instructions:
            object_ids = instruction["object_ids"]  # Object IDs specified in the instruction
            description = instruction["description"]  # Associated description

            # Ensure object_ids is flattened into a 1D list
            if isinstance(object_ids, list):
                object_ids = [item for sublist in object_ids for item in sublist] if any(isinstance(i, list) for i in object_ids) else object_ids
            object_ids = np.array(object_ids).flatten()  # Convert to a flat NumPy array

            # Generate a binary mask for the points that match the object IDs
            mask_point = np.isin(instance_labels.numpy(), object_ids).astype(np.float32)
            # print(f"mask_point.shape: {mask_point.shape}, mask_point.sum: {mask_point.sum()}")

            object_id_tensors.append(torch.tensor(object_ids))
            sampled_sents.append(description)
            sampled_mask_point.append(mask_point.copy())

        # Stack masks and prepare offsets
        masks_points = torch.from_numpy(np.stack(sampled_mask_point, axis=0))
        # print(f"mask_points shape: {masks_points.shape}")
        offset_list = [0]
        for i in range(len(instructions)):
            offset_list.append(offset_list[-1] + 1)  # One mask per instruction
        offset_list = torch.LongTensor(offset_list)

        return {
            "offset": offset_list,
            "masks_points": masks_points,
            "object_id_tensors": object_id_tensors,
            "descriptions": sampled_sents,
        }



    def __getitem__(self, idx):
        """
        Load the processed .npy file, generate 3D masks, and return data for the model.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: Contains processed point cloud data and related ground truth.
        """
        # Load the .npy file
        # Update path for .npy file based on phase (train/validation/test)
        base_dir = "/l/users/jiaxin.huang/jiaxin/dataset/scannet/prepocessed/validation"
        phase_dir = "validation" if self.phase == "validation" else "train"
        
        # Remove 'scene' prefix from the file name
        scene_name = self.scannet_file_list[idx]
        if scene_name.startswith("scene"):
            scene_name = scene_name.replace("scene", "")

        path = os.path.join(
            base_dir,
            # phase_dir,
            scene_name + ".npy",  # Path to the processed .npy file
        )
        data = np.load(path)  # Load point cloud data
        coords = data[:, :3]  # First three columns are x, y, z coordinates
        feats = data[:, 3:6]  # Next three columns are RGB values
        instance_labels = torch.tensor(data[:, -1], dtype=torch.int64)  # Instance labels (last column)
        # print(data.shape)
        # print(data[:5])  # 打印前 5 行检查列顺序

        
        # Normalize features
        feats = torch.tensor(feats, dtype=torch.float32) / 127.5 - 1

        # Normalize and voxelize coordinates
        coords = (coords - coords.mean(axis=0)) / self.voxel_size
        coords = torch.tensor(coords, dtype=torch.float32)
        pc = coords.clone()

        if self.transforms:
            coords = self.transforms(coords)

        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords.contiguous(), return_index=True, return_inverse=True
        )

        # Apply voxelization and subsampling
        feats = feats[indexes]
        # coords = discrete_coords

        # Generate ground truth 3D masks
        scene_name = self.scannet_file_list[idx]
        gt_data = self.get_3D_mask(scene_name, instance_labels)

        # Pack data for model input and ground truth
        packages = (
            pc,
            discrete_coords,
            feats,
            gt_data,
            inverse_indexes,
            self.scannet_file_list[idx],
        )
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
    dset = scannet_Dataset(phase=phase, transforms=transforms, config=config)
    collate_fn = scannet_collate_pair_fn
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
