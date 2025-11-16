import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from torch.utils.data import Dataset
# import pc_utils
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
        coords,
        feats,
        # labels,
        imgs,
        pairing_points,
        pairing_images,
        inverse_indexes,
        scan_names,
        descriptions,
    ) = list(zip(*batch))

    offset_point = 0
    offset_image = 0

    for batch_id in range(len(coords)):
        pairing_points[batch_id][:] += offset_point
        offset_point += coords[batch_id].shape[0]

        pairing_images[batch_id][:, 0] += offset_image
        offset_image += imgs[batch_id].shape[0]

    coords = ME.utils.batched_coordinates(coords, dtype=torch.float32)
    feats = torch.cat(feats, dim=0)
    imgs = torch.cat(imgs, dim=0)
    # print(f"imgs_batch:{imgs.shape}")

    pairing_points = torch.cat(pairing_points, dim=0)
    pairing_images = torch.cat(pairing_images, dim=0)

    descriptions = [desc for batch_descriptions in descriptions for desc in batch_descriptions]

    return {
        "sinput_C": coords,
        "sinput_F": feats,
        "input_I": imgs,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "inverse_indexes": inverse_indexes,
        "descriptions": descriptions,
    }

class scannet_Dataset(Dataset):
    def __init__(self, phase, config, shuffle = True, image_transforms = None, cloud_transforms = None, mixed_transforms = None):

        self.scannet_root_dir = config['dataRoot_scannet']
        # if phase == 'train':
        #     self.scannet_file_list = self.read_files(config['train_file'])
        #     # self.scannet_file_list = self.read_files(config['val_file'])
        # else:
        #     self.scannet_file_list = self.read_files(config['val_file'])


        self.image_transforms = image_transforms
        self.mixed_transforms = mixed_transforms

        self.voxel_size = config['voxel_size']
        self.phase = phase
        self.config = config
        self.imageDim = (640, 480)
        # self.imageDim = (224, 416)
        self.cloud_transforms = cloud_transforms
        self.maxImages = 2
        self.maxPrompts = 2

        # train val在一个 JSON文件中
        # self.description_json_path = config.get('description_json', None)
        # if self.description_json_path is None:
        #     raise KeyError("config中未找到 'description_json' 键，请在配置文件中添加描述文件的路径。")
        # if not os.path.isfile(self.description_json_path):
        #     raise FileNotFoundError(f"描述文件不存在: {self.description_json_path}")
        # with open(self.description_json_path, 'r') as f:
        #     self.all_description_data = json.load(f)

        # 根据 phase 选择加载对应的描述文件
        if phase == 'train':
            desc_path = config.get('description_json_train', None)
            if desc_path is None:
                raise KeyError("config中未找到 'description_json_train' 键，请在配置文件中添加训练描述文件的路径。")
        else:
            desc_path = config.get('description_json_val', None)
            if desc_path is None:
                raise KeyError("config中未找到 'description_json_val' 键，请在配置文件中添加验证描述文件的路径。")

        if not os.path.isfile(desc_path):
            raise FileNotFoundError(f"描述文件不存在: {desc_path}")
        with open(desc_path, 'r') as f:
            self.all_description_data = json.load(f)

        # 直接从描述文件中提取所有独一无二的场景ID作为数据集场景列表
        self.scannet_file_list = list({entry['scene_id'] for entry in self.all_description_data})
        
        # 如果需要还可以对场景列表做排序（可选）
        self.scannet_file_list.sort()

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

        # occlusion_mask = np.abs(depth[np.round(pi[1][inside_mask]).astype(np.int), np.round(pi[0][inside_mask]).astype(np.int)] - p[2][inside_mask]) <= link_proj_threshold
        occlusion_mask = np.abs(depth[np.round(pi[1][inside_mask]).astype(int), np.round(pi[0][inside_mask]).astype(int)] - p[2][inside_mask]) <= link_proj_threshold

        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1
        return link.T

    def __getitem__(self, idx):
        path = os.path.join(self.scannet_root_dir, self.scannet_file_list[idx], self.scannet_file_list[idx]+"_new_semantic.npy")

        data = torch.from_numpy(np.load(path))
        # print(data.shape)
        coords, feats, labels = data[:, :3], data[:, 3: 6], data[:, 9:]
        sceneName = self.scannet_file_list[idx]
        description_data = [entry for entry in self.all_description_data if entry['scene_id'] == sceneName]
        if not description_data:
            raise ValueError(f"未找到场景 {sceneName} 对应的描述数据。")
    
        # # description_path = os.path.join('/home/jiaxin.huang/jiaxin/description/')
        # description_file = f"{sceneName}.json"  # 根据 sceneName 动态获取文件
        # description_path = os.path.join('/home/jiaxin.huang/jiaxin/description/', description_file)

        # if not os.path.isfile(description_path):
        #     raise FileNotFoundError(f"Description file not found: {description_path}")

        # with open(description_path, 'r') as f:
        #     description_data = json.load(f)

        feats = feats / 127.5 - 1

        frame_names = []
        imgs = []
        links = []

        # print(1)

        intrinsic_color = self.read_intrinsic_file(os.path.join(self.config['dataRoot_images'], sceneName, 'intrinsics_color.txt'))
        intrinsic_depth = self.read_intrinsic_file(os.path.join(self.config['dataRoot_images'], sceneName, 'intrinsics_depth.txt'))

        for framename in os.listdir(os.path.join(self.config['dataRoot_images'], sceneName, 'color')):
            frame_names.append(framename.split('.')[0])

        pairing_points = []
        pairing_images = []

        # print(2)

        frame_names = random.sample(frame_names, min(self.maxImages, len(frame_names)))
        descriptions_sampled = random.sample(description_data, min(self.maxPrompts, len(description_data)))
        descriptions = [desc['description'] for desc in descriptions_sampled]

        if len(descriptions) < self.maxImages:
            descriptions = descriptions * (self.maxImages // len(descriptions)) + descriptions[:self.maxImages % len(descriptions)]
            descriptions = descriptions[:self.maxImages]


        for i, frameid in enumerate(frame_names):
            f = os.path.join(self.config['dataRoot_images'], sceneName, 'color', frameid + '.jpg')
            img = imageio.imread(f)
            # img = cv2.imread(f) # Read original image in BGR format
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(f"SceneName: {sceneName}")
            # cv2.imwrite('visual/' + sceneName + frameid + '_before.jpg', img)

            if self.image_transforms:
                # img, applied_transforms = self.image_transforms(img)
                img = self.image_transforms(img)

            # cv2.imwrite('visual/' + sceneName + frameid + '_after.jpg', img)


            img = cv2.resize(img / 255, self.imageDim)
            # print(f"img shape: {img.shape}")
            depth = imageio.imread(f.replace('color', 'depth').replace('.jpg', '.png')) / 1000.0  # convert to meter
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = self.read_pose_file(posePath)

            link = self.computeLinking(pose, coords, depth, 0.05, intrinsic_color, intrinsic_depth, depth.shape)

            pairing_point = torch.from_numpy(np.argwhere(link[:, 2] == 1)).squeeze()
            pairing_points.append(pairing_point)

            link = torch.from_numpy(link).int()
            imgs.append(torch.from_numpy(img.transpose((2, 0, 1))))
            # print(f"imgs shape: {imgs.shape}")
            # breakpoint()

            pairing_image = link[pairing_point, :2]
            pairing_images.append(torch.cat((torch.ones(pairing_point.shape[0], 1) * i,
                                            pairing_image), dim=1))
            

        # print(3)
        # if self.image_transforms:
        #     imgs = self.image_transforms(imgs)

        imgs = torch.stack(imgs)
        # print(f"imgs shape: {imgs.shape}") imgs shape: torch.Size([8, 3, 480, 640])
        pairing_points = torch.cat(pairing_points, dim=0).numpy()
        pairing_images = torch.cat(pairing_images, dim=0).numpy()
        # pairing_points = torch.cat(pairing_points, dim=0)
        # pairing_images = torch.cat(pairing_images, dim=0)

        if self.cloud_transforms:
            coords = self.cloud_transforms(coords.float())


        # print(4)

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

        coords, feats, imgs, pairing_points, pairing_images = coords_b, feats_b, imgs_b, torch.from_numpy(pairing_points_b),\
            torch.from_numpy(pairing_images_b)
        # pairing_points, pairing_images = torch.from_numpy(pairing_points),torch.from_numpy(pairing_images)

        # Normalize and voxelize coordinates
        # coords = (coords - coords.mean(axis=0)) / self.voxel_size
        # coords = torch.tensor(coords, dtype=torch.float32)

        # print(5)

        coords = (coords - coords.mean(0)) / self.voxel_size

        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        # print(6)print

        feats = feats[indexes]
        assert pairing_points.shape[0] == pairing_images.shape[0]
        
        packages = (discrete_coords, 
                    feats, 
                    imgs, 
                    pairing_points, 
                    pairing_images, 
                    inverse_indexes, 
                    self.scannet_file_list[idx], 
                    descriptions)
        return packages
