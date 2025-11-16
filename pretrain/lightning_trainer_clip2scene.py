import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.chamfer_distance import ComputeCDLoss
from pretrain.criterion import NCELoss, DistillKL, semantic_NCELoss
from pytorch_lightning.utilities import rank_zero_only
# from torchsparse import SparseTensor as spvcnn_SparseTensor
from torch import nn
import torch.nn.functional as F
import random
import numba as nb
from utils.scannet_utils import create_color_palette as create_color_palette
from utils.zero_shot_setting import ScanNet_setting as nuScenes_setting

import matplotlib.pyplot as plt


@nb.jit()
def nb_pack(counts):
    return [np.array(list(range(i))) for i in counts]


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

def visual_prediction(coords_scannet, feats_scannet, labels_scannet, predictions, prediction_pre, prediction_CLIPSAMs, lidar_name):

    # import pdb
    # pdb.set_trace()

    # coords_scannet = coords_scannet[:, 1:]
    random_samples = torch.randperm(coords_scannet.size()[0])
    sample_points = 200000

    predictions = predictions[random_samples[:sample_points]].long()
    prediction_pre = prediction_pre[random_samples[:sample_points]].long()
    prediction_CLIPSAMs = prediction_CLIPSAMs[random_samples[:sample_points]].long()
    coords_scannet = coords_scannet[random_samples[:sample_points]]
    feats_scannet = feats_scannet[random_samples[:sample_points]]
    labels_scannet = labels_scannet[random_samples[:sample_points]].long()

    label2color = torch.zeros(coords_scannet.size()).long()
    pred2color = torch.zeros(coords_scannet.size()).long()
    pred_pre2color = torch.zeros(coords_scannet.size()).long()
    pred_CLIPSAM2color = torch.zeros(coords_scannet.size()).long()
    input = torch.zeros(coords_scannet.size()).long()

    # heatmap = np.uint8(255*predictions.detach().cpu().numpy())
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # print(feats_scannet)

    color_template = create_color_palette()

    r = 0 * predictions
    g = 0 * predictions
    b = 0 * predictions

    # index = labels_scannet[:, 1].long() == scannet_utils.CLASS_LABELS.index(object_visual)

    for i in range(20):
        index_label = labels_scannet == i
        index_pred = predictions == i
        index_pred_pre = prediction_pre == i
        index_pred_CLIPSAMs = prediction_CLIPSAMs == i
        if index_label.sum() > 0:
            label2color[index_label] = torch.tensor(color_template[i]).long()
        if index_pred.sum() > 0:
            pred2color[index_pred] = torch.tensor(color_template[i]).long()
        if index_pred_pre.sum() > 0:
            pred_pre2color[index_pred_pre] = torch.tensor(color_template[i]).long()
        if index_pred_CLIPSAMs.sum() > 0:
            pred_CLIPSAM2color[index_pred_CLIPSAMs] = torch.tensor(color_template[i]).long()

    if not os.path.exists('visual_result_scannet_video'): os.makedirs('visual_result_scannet_video')

    write_ply_rgb(coords_scannet, label2color, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'GT'), text=True)
    write_ply_rgb(coords_scannet, pred2color, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'ours'), text=True)
    # write_ply_rgb(coords_scannet, pred_CLIPSAM2color, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'CLIPSAM'), text=True)
    write_ply_rgb(coords_scannet, feats_scannet, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'origin'), text=True)
    # write_ply_rgb(coords_scannet, input, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'origin'), text=True)
    # write_ply_rgb(coords_scannet, pred_pre2color, 'visual_result_scannet_video/%s_%s.ply' % (lidar_name, 'clip'), text=True)


# image, mask_cliporig, seeds_index
def show_anns(img, clip_orig, seeds_index):

    img[img > 1] = 1
    img = img.cpu().contiguous().numpy().astype('float32')
    rand_color = (np.random.rand(10000, 3) * 255).astype(int)
    img_vis = np.zeros_like(img)
    # num_masks = len(mask_data['segmentation'])
    image_name = "kkk"

    # print("1: ", img.min(), img.max())

    # import pdb
    # pdb.set_trace()

    plt.imsave("visual_img/%s.png" % seeds_index, img)

    # for i, mask in enumerate(masks):
    #     seg_i = mask['segmentation']
    #     seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
    #     img_vis += seg_color_map
    #
    # img_vis = np.clip(img_vis, 0, 255)
    # img_vis = img_vis / 255.0
    # img_vis = img_vis * 0.35 + img * 0.65

    # plt.imsave("visual/%s_sam.png" % (image_name), img_vis)

    color_template = create_color_palette()
    # rand_color = color_template

    # img_vis = np.zeros_like(img)
    # img_vis_label = np.zeros_like(img)
    img_vis_oriclip = np.zeros_like(img)
    # img_vis_deeplab = np.zeros_like(img)

    for i, mask in enumerate(clip_orig):
        # seg_i = mask['segmentation']

        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis += seg_color_map
        #
        # seg_i = mask_labels[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_label += seg_color_map
        #
        # seg_i = clip_orig[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_oriclip += seg_color_map
        #
        # seg_i = deeplab[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_deeplab += seg_color_map

        # seg_i = mask['segmentation']
        # img_vis[seg_i] = color_template[i]

        # seg_i = mask_labels[i]['segmentation']
        # img_vis_label[seg_i] = color_template[i]

        seg_i = clip_orig[i]['segmentation']
        img_vis_oriclip[seg_i] = color_template[i]

        # seg_i = deeplab[i]['segmentation']
        # img_vis_deeplab[seg_i] = color_template[i]

    # img_vis = np.clip(img_vis, 0, 255)
    # img_vis = img_vis / 255.0
    # img_vis = img_vis * 0.35 + img * 0.65
    # plt.imsave("visual/%s_clip.png" % (image_name), img_vis)

    #logical_not()
    # img_vis[~(seeds_index.cpu())] = 0
    # img_vis[(seeds_index.cpu())] = img[(seeds_index.cpu())]
    # plt.imsave("visual/%s_seeds.png" % (image_name), img_vis)

    # img_vis_label = np.clip(img_vis_label, 0, 255)
    # img_vis_label = img_vis_label / 255.0
    # img_vis_label = img_vis_label * 0.35 + img * 0.65
    # plt.imsave("visual/%s_label.png" % (image_name), img_vis_label)

    img_vis_oriclip = np.clip(img_vis_oriclip, 0, 255)
    img_vis_oriclip = img_vis_oriclip / 255.0
    img_vis_oriclip = img_vis_oriclip * 0.35 + img * 0.65

    print("1: ", img_vis_oriclip.max(), img_vis_oriclip.min())

    plt.imsave("visual_img/%s_orgclip.png" % (seeds_index), img_vis_oriclip)

    # img_vis_deeplab = np.clip(img_vis_deeplab, 0, 255)
    # img_vis_deeplab = img_vis_deeplab / 255.0
    # img_vis_deeplab = img_vis_deeplab * 0.35 + img * 0.65
    # plt.imsave("visual/%s_deeplab.png" % (image_name), img_vis_deeplab)

    # img_vis_oriclip[~(seeds_index.cpu())] = 0
    # img_vis[(seeds_index.cpu())] = img[(seeds_index.cpu())]
    # plt.imsave("visual/%s_seeds_oriclip.png" % (image_name), img_vis_oriclip)

def visual_masks(image, clip_orig, seeds_index):

    mask_clip = []
    mask_labels = []
    mask_cliporig = []
    mask_deeplab = []

    for i in range(21):
        # index = output_images == i
        # mask_clip.append({'segmentation': index.detach().cpu().numpy()})

        # index = img_labels == i
        # mask_labels.append({'segmentation': index.detach().cpu().numpy()})

        index = clip_orig == i
        mask_cliporig.append({'segmentation': index.detach().cpu().numpy()})

        # index = deeplab_pred == i
        # mask_deeplab.append({'segmentation': index.detach().cpu().numpy()})

    show_anns(image, mask_cliporig, seeds_index)


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, model_fusion, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self.model_fusion = model_fusion
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        self.cot = 0
        self.CE = nn.CrossEntropyLoss()
        self.CD_loss = ComputeCDLoss()
        self.KLloss = DistillKL(T=1)
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.sem_NCE = semantic_NCELoss(temperature=config["NCE_temperature"])


        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        self.text_embeddings_path = config['text_embeddings_path']
        text_categories = config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        if config["mode"] == 'zero_shot':
            if config["dataset"] == "nuscenes":
                self.zero_shot_setting = nuScenes_setting
            if config["dataset"] == "scannet":
                self.zero_shot_setting = ScanNet_setting


        self.saved = False
        self.max_size = 8
    def get_in_field(self, coords, feats):
        in_field = ME.TensorField(coordinates=coords.float(), features=feats.int(),
                                  # coordinate_map_key=A.coordiante_map_key, coordinate_manager=A.coordinate_manager,
                                  quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                  minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                  # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                                  # device=self.config.device,
                                  ).float()
        return in_field


    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()) + list(self.model_fusion.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        # scheduler = optim.lr_scheduler._LRScheduler(optimizer)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        # print(scheduler)
        # print(isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR))
        # print("=============================================================================")
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # optimizer.zero_grad(set_to_none=True)
        # optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        self.model_points.train()

        sinput_C = batch["sinput_C"]
        sinput_F = batch["sinput_F"]

        if self._config['dataset'] == "nuscenes":
            sweepIds = batch["sweepIds"]

        if self._config['max_sweeps'] > 1:
            for sweepid in range(1, self._config['max_sweeps']):
                sweepInd = sweepIds == sweepid
                sinput_C[sweepInd, 0] = sinput_C[sweepInd, 0] + self._config['batch_size'] * sweepid

        if self._config['dataset'] == "scannet":
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
        else:
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
            # sparse_input = spvcnn_SparseTensor(sinput_F, sinput_C)

        output_points = self.model_points(sparse_input)
        output_images = self.model_images(batch["input_I"].float())


        # image_feats, image_pred = output_images

        # for id in range(batch["input_I"].shape[0]):
        #     image = batch["input_I"][id].permute(1, 2, 0).float()
        #     clip_pred = image_pred[id]
        #     visual_masks(image, clip_pred, id)

        # import pdb
        # pdb.set_trace()
        # for id in range(batch["input_I"].shape[0]):
        #     image = batch["input_I"][id].permute(1, 2, 0).float()

        # pairing_images

        del batch["sinput_F"]
        del batch["sinput_C"]
        del batch["input_I"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )

        if not self.saved:

            if self.epoch == 10:
                self.save()
                self.saved = True

        self.train_losses.append(loss.detach().cpu())
        return loss


    def scannet_loss(self, batch, output_points, output_images):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])
        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]

        image_feats, image_pred = output_images
        point_feats_a, point_feats_b = output_points

        # global
        point_logists = F.conv1d(point_feats_a.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
        k_logists = point_logists[pairing_points]
        m_pred = tuple(pairing_images.T.long())
        q_pred = image_pred[m_pred]

        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: q_pred = k_logists.argmax(dim=1)

        loss_semantic = self.CE(k_logists, q_pred)

        point_feats_b = point_feats_b[pairing_points]
        image_feats = image_feats.permute(0, 2, 3, 1)[m_pred]
        loss_spatial = torch.mean(1 - F.cosine_similarity(image_feats, point_feats_b, dim=1))

        return loss_semantic + loss_spatial


    def feature_packaging(self, image_global_allpoints, point_global_allpoints, inverse_indexes_merged, image_pred):
        uni_feature = torch.cat((image_global_allpoints, point_global_allpoints, image_pred.unsqueeze(-1)), dim=1)
        max_inverse_indexes = inverse_indexes_merged.max()
        feature_packages = torch.zeros((max_inverse_indexes + 1) * self.max_size, uni_feature.shape[1]).cuda()

        sorted_inverse_indexes, sorted_indices = torch.sort(inverse_indexes_merged)
        uni_feature = uni_feature[sorted_indices]
        _, counts = torch.unique(sorted_inverse_indexes, return_counts=True)

        offset = nb_pack(counts.detach().cpu().numpy())
        offset = torch.from_numpy(np.concatenate(offset, axis=0)).cuda()
        valid_index = offset < self.max_size

        offset = offset[valid_index]
        sorted_inverse_indexes = sorted_inverse_indexes[valid_index]
        uni_feature = uni_feature[valid_index]

        index = sorted_inverse_indexes * self.max_size + offset
        feature_packages[index] = uni_feature
        feature_packages = feature_packages.view((max_inverse_indexes + 1), self.max_size, uni_feature.shape[1])

        return feature_packages

    def loss_nuscenes(self, batch, output_points, output_images):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        inverse_indexes_group = batch["inverse_indexes_group"]
        inverse_indexes_merged = batch['inverse_indexes_merged']

        image_global, image_pred = output_images
        point_global, point_local = output_points

        point_local = point_local[inverse_indexes_group]
        point_local_allpoints = point_local[pairing_points]

        point_global = point_global[inverse_indexes_group]
        point_global_allpoints = point_global[pairing_points]
        inverse_indexes_merged = inverse_indexes_merged[pairing_points]

        m_pred = tuple(pairing_images.T.long())
        image_global_allpoints = image_global.permute(0, 2, 3, 1)[m_pred]
        image_pred = image_pred[m_pred]

        feature_packages = self.feature_packaging(image_global_allpoints, point_local_allpoints, inverse_indexes_merged, image_pred)

        super_nodes_points, inner_products, pixel_pred = self.model_fusion(feature_packages)
        super_nodes_logit = F.conv1d(point_global_allpoints.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
        loss_semantic = 0

        # Switchable Self-training Strategy
        if self.epoch > 10:
            index_set = set(np.array(list(range(inverse_indexes_group.shape[0]))))
            pairing_set = set(pairing_points.detach().long().cpu().numpy())
            index_set_rest = list(index_set - pairing_set)
            point_global_rest = point_global[index_set_rest]
            point_global_logits = F.conv1d(point_global_rest.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
            point_global_pred = point_global_logits.argmax(dim=1)
            loss_semantic += self.CE(point_global_logits, point_global_pred)

            rd = random.randint(1, 10)
            if rd > 5: image_pred = super_nodes_logit.argmax(dim=1)

        loss_semantic = self.CE(super_nodes_logit, image_pred)
        loss_spatial_temporal = torch.mean(1 - inner_products)
        # loss_spatial_temporal = 0

        return loss_semantic + loss_spatial_temporal

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]

        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q)

    def on_train_epoch_end(self):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):

        sinput_C = batch["sinput_C"]
        sinput_F = batch["sinput_F"]

        if self._config['dataset'] == "nuscenes":
            sweepIds = batch["sweepIds"]
            if self._config['max_sweeps'] > 1:
                for sweepid in range(1, self._config['max_sweeps']):
                    sweepInd = sweepIds == sweepid
                    sinput_C[sweepInd, 0] = sinput_C[sweepInd, 0] + self._config['batch_size'] * sweepid

        if self._config['dataset'] == "scannet":
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
        else:
            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())
            # sparse_input = spvcnn_SparseTensor(sinput_F, sinput_C)

        output_points = self.model_points(sparse_input)

        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "model_fusion": self.model_fusion.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
