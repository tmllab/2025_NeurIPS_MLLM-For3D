import os
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
# from torchsparse import SparseTensor
from utils.metrics import compute_IoU
from torch.nn import functional as F
from PIL import Image
from downstream.model_builder import initialize_lisa_model
import matplotlib.pyplot as plt



def compute_iou(predicted_mask, ground_truth_mask):
    # binary mask 0 1 
    predicted_mask = (predicted_mask > 0.5).long()
    ground_truth_mask = ground_truth_mask.long()

    intersection = (predicted_mask & ground_truth_mask).sum().float()
    union = (predicted_mask | ground_truth_mask).sum().float()

    if union == 0:
        return 1.0  # if both mask ==0 then IoU = 1
    else:
        return intersection / union


def evaluate(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model_images = initialize_lisa_model(config)
    # Ensure the model is moved to the correct device
    model_images.to(device)
    
    total_iou = 0
    total_questions = 0

    with torch.no_grad():
        q = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            scene_names = batch["lidar_name"]
            # inst_labels = batch["inst_labels"]
            len_batch = batch["len_batch"]  # List of lengths, one per sample
            inverse_indexes_list = batch['inverse_indexes']  # List of tensors, one per sample
            offset_list = batch['offset_list']  # Indicates instruction indices per scene
            masks_points = batch['masks_points'].to(device)
            print(scene_names)

            sparse_input = SparseTensor(batch["sinput_F"].float(), batch["sinput_C"].int(), device=0)
            feat_F = model(sparse_input) # shape (M, 512)
            # output_points = model(sparse_input, text_embeddings=text_embedding)
            # print(sparse_input, model)
            # 遍历批次中的每个场景
            
            # offset = 0
            # Compute point offsets for each scene
            point_offsets = [0] + np.cumsum(len_batch).tolist()

            # Initialize instruction offset
            instruction_offset = 0
            offset = 0 # invers
            for idx in range(len(scene_names)):
                scene_name = scene_names[idx]

                # Get start and end indices for points in this scene
                point_start = point_offsets[idx]
                point_end = point_offsets[idx + 1]
                N_scene = point_end - point_start

                # Get the inverse indexes for this scene
                inverse_indexes = inverse_indexes_list[idx].to(device)

                # Map the features back to original points
                scene_feat_F = feat_F[inverse_indexes + point_start]  # Shape: (N, 512)

                # Get instructions for this scene
                num_instructions = offset_list[idx + 1] - offset_list[idx]
                scene_masks_points = masks_points[instruction_offset:instruction_offset + num_instructions].to(device)
                scene_descriptions = batch["descriptions"][instruction_offset:instruction_offset + num_instructions]
                scene_image_filenames = batch["image_filenames"][instruction_offset:instruction_offset + num_instructions]

                # Get text embeddings for all instructions in this scene
                # text_embeddings = []
                # for object_name in scene_object_names:
                #     text_embedding = get_text_embedding(object_name, tokenizer, text_encoder)
                #     text_embeddings.append(text_embedding)
                # text_embeddings = torch.stack(text_embeddings, dim=0)  # Shape: (num_instructions, 512)
                batch_images = []

                for i in range(num_instructions):
                    description = scene_descriptions[i]
                    image_filename = scene_image_filenames[i]
                    image_path = os.path.join(
                        config["dataRoot_scannetpp"], 'data', scene_name, 'dslr', 'undistorted_images', image_filename
                    )

                    # Load the image using PIL
                    image = Image.open(image_path).convert('RGB')
                    image_np = np.array(image)  # Convert to numpy array

                    batch_images.append((image_np, description))
                    # print(batch_images)

                # Get the pred_embeddings from ModelImagesWrapper
                outputs = model_images(batch_images)
                # print(outputs)
                if outputs is None:
                    continue  # Handle this case as needed

                text_embeddings = outputs['pred_embeddings']  # Shape: [num_instructions, embedding_dim]

                # # 对 scene_feat_F 进行归一化
                # scene_feat_F_norm = F.normalize(scene_feat_F, p=2, dim=1)  # Shape: [N_scene, 512]

                # # 对 text_embeddings 进行归一化
                # text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)  # Shape: [num_instructions, 512]

                # # 计算相似度矩阵
                # out = torch.matmul(text_embeddings_norm, scene_feat_F_norm.T)  # Shape: [num_instructions, N_scene]

                # 设置阈值，例如 0.5
                # threshold = 0.7
                # predicted_mask = (out > threshold).long()

                print('text_embeddings:', text_embeddings)
                print(f'feat_F shape: {feat_F.shape}')          # 应该是 (N_total_points, 512)
                print(f'text_embeddings shape: {text_embeddings.shape}')  # 应该是 (num_instructions, 512)
                out = F.conv1d(scene_feat_F.unsqueeze(-1), text_embeddings.unsqueeze(-1)).squeeze()
                # print(out)
                out = out.T 
                print(out)
                print(f'scene_feat_F mean: {scene_feat_F.mean().item()}')
                print(f'scene_feat_F std: {scene_feat_F.std().item()}')
                print(f'text_embeddings mean: {text_embeddings.mean().item()}')
                print(f'text_embeddings std: {text_embeddings.std().item()}')

                
                # threshold = 0.0
                # predicted_mask = (out > threshold).long()
                # Compute IoU for each instruction
                # for i in range(num_instructions):
                #     predicted_mask = (out > 0).long()

                #     print(f'out[i] shape: {out[i].shape}')
                #     print('out[i]:', out[i])
                #     print('out[i] mean:', out[i].mean().item())
                #     print('out[i] std:', out[i].std().item())
                #     print(predicted_mask)
                #     print('Sum of predicted_mask:', predicted_mask.sum().item())
                #     ground_truth_mask = scene_masks_points[i]
                #     # print(f'ground_truth_mask shape: {ground_truth_mask.shape}')
                #     # print('ground_truth_mask:', ground_truth_mask)
                #     # print('Sum of ground_truth_mask:', ground_truth_mask.sum().item())
                #     # predicted_mask_i = predicted_mask[i]
                #     # ground_truth_mask_i = scene_masks_points[i]
                #     # iou = compute_iou(predicted_mask_i, ground_truth_mask_i)

                #     iou = compute_iou(predicted_mask, ground_truth_mask)
                #     total_iou += iou
                #     total_questions += 1

                #     print(f'Scene: {scene_name}, Instruction {i+1}/{num_instructions}, IoU: {iou:.4f}')
                for i in range(num_instructions):
                    # 计算统计量
                    out_i = out[i]
                    mean = out_i.mean().item()
                    std = out_i.std().item()
                    max_value = out_i.max().item()
                    min_value = out_i.min().item()
                    median_value = out_i.median().item()
                    k=0.5

                    print(f'out[{i}] mean: {mean}')
                    print(f'out[{i}] std: {std}')
                    print(f'out[{i}] max: {max_value}')
                    print(f'out[{i}] min: {min_value}')
                    print(f'out[{i}] median: {median_value}')

                    # 绘制直方图（可选）
                    # out_i_cpu = out_i.detach().cpu().numpy()
                    # plt.hist(out_i_cpu, bins=100)
                    # plt.title(f'Distribution of out[{i}]')
                    # plt.xlabel('Value')
                    # plt.ylabel('Frequency')
                    # plt.show()

                    # 设置阈值（您可以根据需要调整）
                    # 方案一：使用中位数作为阈值
                    # threshold = median_value
                    

                    # 方案二：使用 mean + k * std
                    k = 0.8
                    threshold = mean + k * std

                    # 方案三：使用固定阈值
                    # threshold = 0.0
                    # threshold = torch.quantile(out_i, 0.9).item()  # 取第 90 个百分位数


                    # 生成预测掩码
                    predicted_mask = (out_i > threshold).long()
                    ground_truth_mask = scene_masks_points[i]
                    # 统计预测的正样本数量
                    num_predicted = predicted_mask.sum().item()
                    num_ground_truth = ground_truth_mask.sum().item()
                    print(f'Number of predicted positives: {num_predicted}')
                    print(f'Number of ground truth positives: {num_ground_truth}')
                    print(f'ground_truth_mask shape: {ground_truth_mask.shape}')
                    print('ground_truth_mask:', ground_truth_mask)

                    # 计算 IoU
                    iou = compute_iou(predicted_mask, ground_truth_mask)
                    total_iou += iou
                    total_questions += 1

                    print(f'Scene: {scene_name}, Instruction {i+1}/{num_instructions}, IoU: {iou:.4f}')

                    # {
                    #     "scene_id": "c4c04e6d6c",
                    #     "object_id": [
                    #         64
                    #     ],
                    #     "object_name": "wall clock",
                    #     "image": "DSC03384.JPG",
                    #     "description": "What item placed above the bookshelf is used to tell the time?"
                    # },
                # Update instruction offset
                instruction_offset += num_instructions

    average_iou = total_iou / total_questions
    print(f'Average IoU on validation set: {average_iou:.4f}')

    # torch.cuda.empty_cache()