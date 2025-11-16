import os
import numpy as np
import torch
from tqdm import tqdm
from MinkowskiEngine import SparseTensor
from utils.metrics import compute_IoU
from torch.nn import functional as F
from PIL import Image
from downstream.model_builder import initialize_lisa_model
import logging

# 初始化日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation_results_30_v23.log"),  # 保存到文件
        logging.StreamHandler()  # 同时打印到控制台
    ]
)

def compute_iou(predicted_mask, ground_truth_mask):
    predicted_mask = (predicted_mask > 0.0).long()
    ground_truth_mask = ground_truth_mask.long()

    intersection = (predicted_mask & ground_truth_mask).sum().float()
    union = (predicted_mask | ground_truth_mask).sum().float()

    if union == 0:
        return 1.0  # If both masks are empty, IoU = 1
    else:
        return intersection / union

def evaluate(model, dataloader, config):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_images = initialize_lisa_model(config)
    model_images.to(device)

    total_iou = 0
    total_questions = 0

    with open("evaluation_results_30_v23.txt", "w") as result_file:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                scene_names = batch["lidar_name"]
                len_batch = batch["len_batch"]
                inverse_indexes_list = batch['inverse_indexes']
                masks_points = batch['masks_points'].to(device)
                num_description = batch["num_description"]

                sparse_input = SparseTensor(batch["sinput_F"].float(), batch["sinput_C"].int(), device=0)
                feat_F = model(sparse_input)

                point_offsets = [0] + np.cumsum(len_batch).tolist()
                instruction_offset = 0

                for idx, scene_name in enumerate(scene_names):
                    point_start = point_offsets[idx]
                    point_end = point_offsets[idx + 1]
                    inverse_indexes = inverse_indexes_list[idx].to(device)
                    scene_feat_F = feat_F[inverse_indexes + point_start]

                    num_instructions = num_description[idx]
                    scene_masks_points = masks_points[instruction_offset:instruction_offset + num_instructions]
                    scene_descriptions = batch["descriptions"][instruction_offset:instruction_offset + num_instructions]

                    max_pred_embeddings = [] 
                    for description in scene_descriptions:
                        max_pixels = 0
                        best_embedding = None
                        best_frame_id = None
                        for frame_id in os.listdir(os.path.join(config['dataRoot_images'], scene_name, 'color')):
                            image_path = os.path.join(config['dataRoot_images'], scene_name, 'color', frame_id)
                            try:
                                image = Image.open(image_path).convert('RGB')
                                image_np = np.array(image)
                                outputs = model_images((image_np, description))
                            except Exception as e:
                                logging.error(f"Error processing {image_path}: {e}")
                                continue

                            if outputs is None or 'image_pred' not in outputs or 'pred_embeddings' not in outputs:
                                continue

                            mask = outputs['image_pred']
                            mask = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask)
                            num_pixels = (mask > 0).sum().item()

                            if num_pixels > max_pixels:
                                max_pixels = num_pixels
                                best_embedding = outputs['pred_embeddings']
                                best_frame_id = frame_id

                        if best_embedding is not None:
                            max_pred_embeddings.append(best_embedding)
                            logging.info(f"Best image for description '{description}': {best_frame_id} with {max_pixels} pixels")
                            result_file.write(f"Best image for description '{description}': {best_frame_id} with {max_pixels} pixels\n")

                    if len(max_pred_embeddings) == 0:
                        logging.warning(f"Scene {scene_name}: No valid embeddings generated for descriptions.")
                        instruction_offset += num_instructions
                        continue

                    for embedding, gt_mask in zip(max_pred_embeddings, scene_masks_points):
                        out = F.conv1d(scene_feat_F.unsqueeze(-1), embedding.unsqueeze(-1)).squeeze().T
                        predicted_mask = (out > 0.0).long()
                        iou = compute_iou(predicted_mask, gt_mask)
                        total_iou += iou
                        total_questions += 1

                        log_msg = f"Scene: {scene_name}, IoU: {iou:.4f}"
                        logging.info(log_msg)
                        result_file.write(log_msg + "\n")

                    instruction_offset += num_instructions

        average_iou = total_iou / total_questions
        logging.info(f'Average IoU on validation set: {average_iou:.4f}')
        result_file.write(f'Average IoU on validation set: {average_iou:.4f}\n')

    torch.cuda.empty_cache()
