import torch
import random
import numpy as np
# from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resize, resized_crop, hflip
import math
from detectron2.data import transforms as T

from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from .augmentation import Augmentation, _transform_to_aug
from detectron2.data.transforms import apply_transform_gens
from fvcore.transforms.transform import Transform
import cv2
class ColorAugSSDTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())
        self.intensity_min = 0.8
        self.intensity_max = 1.2

    def get_transform(self, image):
    # def __call__(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())
        self.intensity_min = 0.8
        self.intensity_max = 1.2


    def get_transform(self, image):
    # def __call__(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)

class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())
        self.intensity_min = 0.8
        self.intensity_max = 1.2


    def get_transform(self, image):
    # def __call__(self, image):
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)




class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
        self.scale = 0.15
    #
    def get_transform(self, image):
    # def __call__(self, image):
        assert image.shape[-1] == 3, "RandomLighting only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )


class ComposeClouds:
    """
    Compose multiple transformations on a point cloud.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc):
        for transform in self.transforms:
            pc = transform(pc)
        return pc

class ComposeImages:
    """
    Compose multiple transformations on a point cloud.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        # img = apply_transform_gens(self.transforms, img)
        for transform in self.transforms:
        #     img = transform(img)
            img = transform.apply_image(img)
        return img

class Rotation_z:
    """
    Random rotation of a point cloud around the z axis.
    """

    def __init__(self):
        pass

    def __call__(self, pc):
        angle = np.random.random() * 2 * np.pi
        c = np.cos(angle)
        s = np.sin(angle)
        R = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        pc = pc @ R.T
        return pc


class FlipAxis:
    """
    Flip a point cloud in the x and/or y axis, with probability p for each.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pc):
        for curr_ax in range(2):
            if random.random() < self.p:
                pc[:, curr_ax] = -pc[:, curr_ax]
        return pc


class random_rotation_scalling_flipping:
    def __init__(self, p=0.5):
        self.p = p


    def __call__(self, coords):
        scale_flip = np.eye(3) + np.random.randn(3, 3) * 0.1
        scale_flip[0][0] *= np.random.randint(0, 2) * 2 - 1
        scale_flip = torch.from_numpy(scale_flip).float()

        # scale = torch.eye(3)
        theta = random.uniform(0, 2) * math.pi
        rotationx = torch.tensor([[math.cos(theta), math.sin(theta), 0],
                                 [-math.sin(theta), math.cos(theta), 0],
                                 [0, 0, 1]]).float()

        m = torch.matmul(scale_flip, rotationx)
        coords = torch.matmul(coords.float(), m)
        return coords



def make_transforms_clouds(config):
    """
    Read the config file and return the desired transformation on point clouds.
    """
    transforms = []
    if config["transforms_clouds"] is not None:
        for t in config["transforms_clouds"]:
            if config['dataset'] == 'scannet' and config['mode'] == 'finetune':
                transforms.append(random_rotation_scalling_flipping())
                # print("sssss")
            else:
                if t.lower() == "rotation":
                    transforms.append(Rotation_z())
                elif t.lower() == "flipaxis":
                    transforms.append(FlipAxis())
                else:
                    raise Exception(f"Unknown transformation: {t}")
    if not len(transforms):
        return None
    return ComposeClouds(transforms)

def make_transforms_images(config):
    """
    Read the config file and return the desired transformation on point clouds.
    """
    transforms = []
    if config["transforms_images"] is not None:
        for t in config["transforms_images"]:

            if t.lower() == "coloraugssdtransform":
                transforms.append(ColorAugSSDTransform(img_format="RGB"))
            # elif t.lower() == "randomcontrast":
            #     transforms.append(T.RandomContrast(0.5, 1.5))
            #     # transforms.append(RandomContrast())
            # elif t.lower() == "randombrightness":
            #     transforms.append(T.RandomBrightness(0.8, 1.2))
            #     # transforms.append(RandomBrightness())
            # elif t.lower() == "randomsaturation":
            #     transforms.append(T.RandomSaturation(0.8, 1.2))
            #     # transforms.append(RandomSaturation())
            # elif t.lower() == "randomlighting":
            #     transforms.append(T.RandomLighting(0.15))
                # transforms.append(RandomLighting())
            else:
                raise Exception(f"Unknown transformation: {t}")

    if not len(transforms):
        return None
    return ComposeImages(transforms)


class ComposeAsymmetrical:
    """
    Compose multiple transformations on a point cloud, and image and the
    intricate pairings between both (only available for the heavy dataset).
    Note: Those transformations have the ability to increase the number of
    images, and drastically modify the pairings
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc, features, img, pairing_points, pairing_images, superpixels=None):
        for transform in self.transforms:
            pc, features, img, pairing_points, pairing_images, superpixels = transform(
                pc, features, img, pairing_points, pairing_images, superpixels
            )
        if superpixels is None:
            return pc, features, img, pairing_points, pairing_images
        return pc, features, img, pairing_points, pairing_images, superpixels


class ResizedCrop:
    """
    Resize and crop an image, and adapt the pairings accordingly.
    """

    def __init__(
        self,
        image_crop_size=(224, 416),
        image_crop_range=[0.3, 1.0],
        image_crop_ratio=(14.0 / 9.0, 17.0 / 9.0),
        crop_center=False,
    ):
        self.crop_size = image_crop_size
        self.crop_range = image_crop_range
        self.crop_ratio = image_crop_ratio
        # self.img_interpolation = image_interpolation
        self.crop_center = crop_center

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels=None):
        imgs = torch.empty(
            (images.shape[0], 3) + tuple(self.crop_size), dtype=torch.float32
        )
        if superpixels is not None:
            superpixels = superpixels.unsqueeze(1)
            sps = torch.empty(
                (images.shape[0],) + tuple(self.crop_size), dtype=torch.uint8
            )
        pairing_points_out = np.empty(0, dtype=np.int64)
        pairing_images_out = np.empty((0, 3), dtype=np.int64)
        if self.crop_center:
            pairing_points_out = pairing_points
            _, _, h, w = images.shape
            for id, img in enumerate(images):
                mask = pairing_images[:, 0] == id
                p2 = pairing_images[mask]
                p2 = np.round(
                    np.multiply(p2, [1.0, self.crop_size[0] / h, self.crop_size[1] / w])
                ).astype(np.int64)

                imgs[id] = resize(img, self.crop_size)
                if superpixels is not None:
                    sps[id] = resize(
                        superpixels[id], self.crop_size, InterpolationMode.NEAREST
                    )

                p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                pairing_images_out = np.concatenate((pairing_images_out, p2))

        else:
            for id, img in enumerate(images):
                successfull = False
                mask = pairing_images[:, 0] == id
                P1 = pairing_points[mask]
                P2 = pairing_images[mask]
                while not successfull:
                    i, j, h, w = RandomResizedCrop.get_params(
                        img, self.crop_range, self.crop_ratio
                    )
                    p1 = P1.copy()
                    p2 = P2.copy()
                    p2 = np.round(
                        np.multiply(
                            p2 - [0, i, j],
                            [1.0, self.crop_size[0] / h, self.crop_size[1] / w],
                        )
                    ).astype(np.int64)

                    valid_indexes_0 = np.logical_and(
                        p2[:, 1] < self.crop_size[0], p2[:, 1] >= 0
                    )
                    valid_indexes_1 = np.logical_and(
                        p2[:, 2] < self.crop_size[1], p2[:, 2] >= 0
                    )
                    valid_indexes = np.logical_and(valid_indexes_0, valid_indexes_1)
                    sum_indexes = valid_indexes.sum()
                    len_indexes = len(valid_indexes)
                    # print(len_indexes)
                    if len_indexes == 0: continue
                    if sum_indexes > 1024 or sum_indexes / len_indexes > 0.75:
                        successfull = True
                imgs[id] = resized_crop(
                    img, i, j, h, w, self.crop_size
                )
                if superpixels is not None:
                    sps[id] = resized_crop(
                        superpixels[id],
                        i,
                        j,
                        h,
                        w,
                        self.crop_size,
                    )
                pairing_points_out = np.concatenate(
                    (pairing_points_out, p1[valid_indexes])
                )
                pairing_images_out = np.concatenate(
                    (pairing_images_out, p2[valid_indexes])
                )
        if superpixels is None:
            return pc, features, imgs, pairing_points_out, pairing_images_out, superpixels
        return pc, features, imgs, pairing_points_out, pairing_images_out, sps


class FlipHorizontal:
    """
    Flip horizontaly the image with probability p and adapt the matching accordingly.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels=None):
        w = images.shape[3]
        for i, img in enumerate(images):
            if random.random() < self.p:
                images[i] = hflip(img)
                mask = pairing_images[:, 0] == i
                pairing_images[mask, 2] = w - 1 - pairing_images[mask, 2]

        return pc, features, images, pairing_points, pairing_images, superpixels


class DropCuboids:
    """
    Drop random cuboids in a cloud
    """

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels=None):
        range_xyz = torch.max(pc, axis=0)[0] - torch.min(pc, axis=0)[0]

        crop_range = np.random.random() * 0.2
        new_range = range_xyz * crop_range / 2.0

        sample_center = pc[np.random.choice(len(pc))]

        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = torch.sum((pc[:, 0:3] < max_xyz).to(torch.int32), 1) == 3
        lower_idx = torch.sum((pc[:, 0:3] > min_xyz).to(torch.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        pc_out = pc[new_pointidx]
        features_out = features[new_pointidx]

        mask = new_pointidx[pairing_points]
        cs = torch.cumsum(new_pointidx, 0) - 1
        pairing_points_out = pairing_points[mask]
        pairing_points_out = cs[pairing_points_out]
        pairing_images_out = pairing_images[mask]

        successfull = True
        for id in range(len(images)):
            if np.sum(pairing_images_out[:, 0] == id) < 1024:
                successfull = False
        if successfull:
            return (
                pc_out,
                features_out,
                images,
                np.array(pairing_points_out),
                np.array(pairing_images_out),
            )
        return pc, features, images, pairing_points, pairing_images, superpixels


def make_transforms_asymmetrical(config):
    """
    Read the config file and return the desired mixed transformation.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                # pass
                transforms.append(
                    ResizedCrop(
                        image_crop_size=config["crop_size"],
                        image_crop_ratio=config["crop_ratio"],
                        crop_center=True,
                    )
                )
            elif t.lower() == "fliphorizontal":
                transforms.append(FlipHorizontal())
            elif t.lower() == "dropcuboids":
                transforms.append(DropCuboids())
            else:
                raise Exception(f"Unknown transformation {t}")
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)


def make_transforms_asymmetrical_val(config):
    """
    Read the config file and return the desired mixed transformation
    for the validation only.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                # pass
                transforms.append(
                    ResizedCrop(image_crop_size=config["crop_size"], crop_center=True)
                )
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)
