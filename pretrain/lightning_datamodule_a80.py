import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pretrain.dataloader_scannetpp import (
    scannetpp_Dataset,
    scannetpp_collate_pair_fn,
)
from pretrain.dataloader_scannet import (
    scannet_Dataset,
    scannet_collate_pair_fn,
)

try:
    from pretrain.dataloader_nuscenes_spconv import NuScenesMatchDatasetSpconv, spconv_collate_pair_fn
except ImportError:
    NuScenesMatchDatasetSpconv = None
    spconv_collate_pair_fn = None
from utils.transforms import (
    make_transforms_images,
    make_transforms_clouds,
    make_transforms_asymmetrical,
    make_transforms_asymmetrical_val,
)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"] // config.get("num_gpus", 1)

    def setup(self, stage=None):
        image_transforms_train = make_transforms_images(self.config)
        cloud_transforms_train = make_transforms_clouds(self.config)
        mixed_transforms_train = make_transforms_asymmetrical(self.config)
        cloud_transforms_val = None
        image_transforms_val = None
        mixed_transforms_val = make_transforms_asymmetrical_val(self.config)

        dataset_name = self.config["dataset"].lower()
        model_points = self.config["model_points"].lower()
        phase_train, phase_val = self._get_phase_names()

        # Select the appropriate dataset
        Dataset = self._select_dataset(dataset_name, model_points)

        # Load training dataset
        self.train_dataset = Dataset(
            phase=phase_train,
            config=self.config,
            shuffle=True,
            image_transforms=image_transforms_train,
            cloud_transforms=cloud_transforms_train,
            mixed_transforms=mixed_transforms_train,
        )
        print("Dataset Loaded")
        print("training size: ", len(self.train_dataset))

        # Load validation dataset
        if dataset_name == "nuscenes":
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                image_transforms=image_transforms_val,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
            )
        print("validation size: ", len(self.val_dataset))

    def _select_dataset(self, dataset_name, model_points):
        if dataset_name == "nuscenes" and model_points == "minkunet":
            return NuScenesMatchDataset
        elif dataset_name == "kitti":
            return KittiMatchDataset
        elif dataset_name == "scannet":
            return scannet_Dataset
        elif dataset_name == "nuscenes" and model_points == "voxelnet":
            return NuScenesMatchDatasetSpconv
        elif dataset_name == "scannetpp":
            return scannetpp_Dataset
        else:
            raise Exception("Dataset Unknown")

    def _get_phase_names(self):
        if self.config["training"] in ("parametrize", "parametrizing"):
            return "parametrizing", "verifying"
        return "train", "val"

    def _get_collate_fn(self, dataset_name):
        if dataset_name == "nuscenes":
            return minkunet_collate_pair_fn
        elif dataset_name == "kitti":
            return kitti_collate_pair_fn
        elif dataset_name == "scannet":
            return scannet_collate_pair_fn
        elif dataset_name == "scannetpp":
            return scannetpp_collate_pair_fn
        else:
            raise Exception("Collate function for dataset unknown")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def _create_dataloader(self, dataset, shuffle, drop_last):
        num_workers = self.config["num_threads"] // self.config.get("num_gpus", 1)
        collate_fn = self._get_collate_fn(self.config["dataset"].lower())

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        )
