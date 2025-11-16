import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from model import (
    SPVCNN,
    MinkUNet,
    # VoxelNet,
    DilationFeatureExtractor,
    PPKTFeatureExtractor,
    Preprocessing,
    DinoVitFeatureExtractor,
    fusionNet,
    # maskClipFeatureExtractor,
)
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, 
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# ---------------- Helper Functions ----------------

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def adjust_precision(tensor, precision):
    """将 tensor 根据 precision 参数转换为对应的数据类型。"""
    if precision == "bf16":
        return tensor.bfloat16()
    elif precision == "fp16":
        return tensor.half()
    else:
        return tensor.float()

# ---------------- GPU Batch Visual Embeddings ----------------

def get_visual_embs(self, pixel_values: torch.FloatTensor):
    """批量获取图像特征，假设 image_encoder 支持批量输入。"""
    with torch.no_grad():
        return self.model.visual_model.image_encoder(pixel_values)

# ---------------- ModelImagesWrapper ----------------

class ModelImagesWrapper(nn.Module):
    """
    一个包装类，用于冻结 LISAForCausalLM 模型并在推理时批量处理输入图像和描述。
    """
    def __init__(self, model, tokenizer, clip_image_processor, transform, config):
        super(ModelImagesWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.config = config
        self.model.eval()  # 固定模型参数，仅用于推理

    def forward(self, batch): 
        """
        Forward 函数：批量处理输入数据。
        Args:
            batch: 字典，包含：
                - input_I: 图像 tensor，形状 [B, C, W, H]
                - descriptions: 每个图像对应的描述（prompt）的列表
        Returns:
            包含预测 mask、图像特征和 [SEG] token 嵌入的字典
        """
        device = next(self.model.parameters()).device
        images = batch['input_I']  # shape: [B, C, W, H]
        B = images.shape[0]
        # 将每个图像转换为 numpy 格式 [W, H, C]，便于预处理
        images_np_list = [img.permute(1, 2, 0).cpu().numpy() for img in images]
        descriptions = batch['descriptions']  # List[str]

        input_ids_list = []
        image_clip_list = []
        image_sam_list = []
        original_size_list = []
        resize_list = []

        # 对每个样本分别预处理
        for i, (image_np, prompt) in enumerate(zip(images_np_list, descriptions)):
            # 构造 prompt（利用对话模板）
            conv_type = self.config.get("conv_type", "llava_v1")
            conv = conversation_lib.conv_templates[conv_type].copy()
            conv.messages = []
            prompt = str(prompt)
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + " Please output segmentation mask."
            if self.config["use_mm_start_end"]:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            full_prompt = conv.get_prompt()

            # Tokenize prompt
            input_ids = tokenizer_image_token(full_prompt, self.tokenizer, return_tensors="pt")
            input_ids_list.append(input_ids.squeeze(0))
            
            # 处理 CLIP 分支的图像
            clip_out = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
            clip_out = clip_out.unsqueeze(0).to(device)
            clip_out = adjust_precision(clip_out, self.config["precision"])
            image_clip_list.append(clip_out)
            
            # 处理 SAM 分支图像
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
            sam_img = preprocess(image_tensor).unsqueeze(0).to(device)
            sam_img = adjust_precision(sam_img, self.config["precision"])
            image_sam_list.append(sam_img)
            
            # 记录原始尺寸和目标尺寸
            original_size = image_np.shape[:2]  # (H, W)
            original_size_list.append(original_size)
            resize_size = sam_img.shape[2:]  # (H, W)
            resize_list.append(resize_size)
        
        # 堆叠成批次
        images_clip = torch.cat(image_clip_list, dim=0)  # [B, C, H, W]
        images_sam = torch.cat(image_sam_list, dim=0)       # [B, C, H, W]
        input_ids_batch = torch.stack(input_ids_list, dim=0).to(device)  # [B, seq_len]

        # 调用模型的 evaluate 函数（假设支持批量输入）
        output_ids, pred_masks, pred_embeddings = self.model.evaluate(
            images_clip=images_clip,
            images=images_sam,
            input_ids=input_ids_batch,
            resize_list=resize_list,
            original_size_list=original_size_list,
            max_new_tokens=32,
            tokenizer=self.tokenizer,
        )

        # 获取图像特征（批量）
        image_features = self.model.get_visual_embs(images_sam)  # [B, C, H', W']

        # 后处理 pred_masks 和 pred_embeddings
        all_pred_masks = []
        all_pred_embeddings = []
        for i in range(len(pred_masks)):
            mask_i = pred_masks[i]
            if mask_i.shape[0] > 0:
                # 这里取第一个 mask（或根据 iou 选择最佳）
                mask_i = mask_i[0]
                mask_i = mask_i.detach().cpu().numpy()
                mask_i = mask_i > 0
                mask_i = torch.from_numpy(mask_i).long().to(device)
                all_pred_masks.append(mask_i)
            else:
                h, w = resize_list[i]
                all_pred_masks.append(torch.zeros((h, w), dtype=torch.long, device=device))
            
            if len(pred_embeddings[i]) > 0:
                emb_i = torch.cat(pred_embeddings[i], dim=0)  # [num_embeddings, embedding_dim]
                if emb_i.dim() == 1:
                    emb_i = emb_i.unsqueeze(0)
                all_pred_embeddings.append(emb_i)
            else:
                all_pred_embeddings.append(torch.zeros((1, self.config["model_n_out"]), device=device))
        
        all_pred_masks = torch.stack(all_pred_masks, dim=0)  # [B, H, W]
        # 对于 pred_embeddings，由于每个样本的个数可能不同，保留列表形式

        return {
            'image_pred': all_pred_masks,
            'image_features': image_features,
            'pred_embeddings': all_pred_embeddings,
        }

# ---------------- make_model 和 initialize_lisa_model ----------------

def make_model(config):
    """
    根据配置构建点云模型、图像模型以及融合模块
    """
    model_fusion = fusionNet(config)
    if config['dataset'] in ["scannet", "scannetpp"]:
        model_points = MinkUNet(3, config["model_n_out"], config)
    else:
        model_points = MinkUNet(1, config["model_n_out"], config)

    if config["images_encoder"] == "lisa":
        model_images = initialize_lisa_model(config)
    elif config["images_encoder"].find("vit_") != -1:
        model_images = DinoVitFeatureExtractor(config, preprocessing=Preprocessing())
    # elif config["images_encoder"] == "maskclip":
    #     model_images = maskClipFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "dilation":
        model_images = DilationFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "ppkt":
        model_images = PPKTFeatureExtractor(config, preprocessing=Preprocessing())
    else:
        raise Exception(f"Model not found: {config['decoder']}")

    return model_points, model_images, model_fusion

def initialize_lisa_model(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["version"],
        cache_dir=None,
        model_max_length=config["model_max_length"],
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    config["seg_token_idx"] = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if config["precision"] == "bf16":
        torch_dtype = torch.bfloat16
    elif config["precision"] == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if config.get("load_in_4bit", False):
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif config.get("load_in_8bit", False):
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        config["version"],
        low_cpu_mem_usage=True,
        vision_tower=config["vision_tower"],
        seg_token_idx=config["seg_token_idx"],
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if config["precision"] == "bf16":
        model = model.bfloat16().cuda()
    elif config["precision"] == "fp16" and (not config.get("load_in_4bit", False)) and (not config.get("load_in_8bit", False)):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif config["precision"] == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=config["local_rank"])

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(config["image_size"])

    model.eval()

    model_lisa = ModelImagesWrapper(
        model=model,
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor,
        transform=transform,
        config=config
    )

    return model_lisa

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# import numpy as np
# from model import (
#     SPVCNN,
#     MinkUNet,
#     # VoxelNet,
#     DilationFeatureExtractor,
#     PPKTFeatureExtractor,
#     Preprocessing,
#     DinoVitFeatureExtractor,
#     fusionNet,
#     # maskClipFeatureExtractor,
# )
# from model.LISA import LISAForCausalLM
# from model.llava import conversation as conversation_lib
# from model.llava.mm_utils import tokenizer_image_token
# from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
# from model.segment_anything.utils.transforms import ResizeLongestSide
# from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, 
#                          DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


# def preprocess(
#     x,
#     pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
#     pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
#     img_size=1024,
# ) -> torch.Tensor:
#     """Normalize pixel values and pad to a square input."""
#     # Normalize colors
#     x = (x - pixel_mean) / pixel_std
#     # Pad
#     h, w = x.shape[-2:]
#     padh = img_size - h
#     padw = img_size - w
#     x = F.pad(x, (0, padw, 0, padh))
#     return x

# def get_visual_embs(self, pixel_values: torch.FloatTensor):
#     with torch.no_grad():
#         image_embeddings_list = []
#         for i in range(pixel_values.shape[0]):
#             torch.cuda.empty_cache()
#             image_embeddings = self.model.visual_model.image_encoder(
#                 pixel_values[i].unsqueeze(0)
#             )
#             image_embeddings_list.append(image_embeddings)
#         torch.cuda.empty_cache()
#         image_embeddings = torch.cat(image_embeddings_list, 0)
#     return image_embeddings


# class ModelImagesWrapper(nn.Module):
#     """
#     A wrapper class for LISAForCausalLM to freeze and use it for inference only.
#     """
#     def __init__(self, model, tokenizer, clip_image_processor, transform, config):
#         super(ModelImagesWrapper, self).__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.clip_image_processor = clip_image_processor
#         self.transform = transform
#         self.config = config
#         self.model.eval()  # Freeze LISA model for inference

#     def forward(self, batch): 
#         """
#         Forward function that performs inference using LISA model.
#         Args:
#             batch: Dictionary containing:
#                 - input_I: imgs from data loader [B, C, W, H]
#                 - descriptions: List of strings with prompts for each image
#         Returns:
#             Dictionary containing:
#                 - image_pred: Predicted masks
#                 - image_features: Extracted image features
#                 - text_embeddings: Extracted embeddings for [SEG] tokens
#         """
#         # Extracting data from batch
#         images = batch['input_I']  # [B, W, H, C] format
#         images = images.permute(0, 2, 3, 1)  # Now shape: [8, 224, 416, 3]
#         descriptions = batch['descriptions']  # List of prompts for each image
#         # print(f"images: {images}")

#         device = next(self.model.parameters()).device
#         # List to collect outputs for each image-description pair
#         all_pred_masks = []
#         all_pred_embeddings = []
#         all_image_features = []
#         # Iterate over images and prompts to perform inference
#         for image_np, prompt in zip(images, descriptions):
#             # Prepare prompt
#             single_sample_embeddings = []
#             image_np = image_np.cpu().numpy()  # [W, H, C]
#             # print(f"image_np[i]: {image_np[i]}")
            
#             conv_type = self.config.get("conv_type", "llava_v1") 
#             conv = conversation_lib.conv_templates[conv_type].copy()
#             conv.messages = []
            
#             prompt = str(prompt)
#             prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + "Please output segmentation mask."
            
#             if self.config["use_mm_start_end"]:
#                 replace_token = (
#                     DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#                 )
#                 prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], "")
#             prompt = conv.get_prompt()

#             # Prepare image data for model input
#             original_size_list = [image_np.shape[:2]]  # Extract original H, W
#             # Process image for CLIP-based input (image_clip)
#             image_clip = (
#                 self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
#                 .unsqueeze(0)
#                 .to(device)
#             )

#             # Adjust precision if needed
#             if self.config["precision"] == "bf16":
#                 image_clip = image_clip.bfloat16()
#             elif self.config["precision"] == "fp16":
#                 image_clip = image_clip.half()
#             else:
#                 image_clip = image_clip.float()

#             # Transform and preprocess the image for SAM input
#             image_sam = (
#                 preprocess(torch.from_numpy(image_np).permute(2, 0, 1).contiguous())
#                 .unsqueeze(0)
#                 .to(device)
#             )

#             # Adjust precision if needed
#             if self.config["precision"] == "bf16":
#                 image_sam = image_sam.bfloat16()
#             elif self.config["precision"] == "fp16":
#                 image_sam = image_sam.half()
#             else:
#                 image_sam = image_sam.float()

#             # Tokenize prompt
#             input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
#             input_ids = input_ids.unsqueeze(0).to(device)

#             # Perform inference using model's evaluate() function
#             output_ids, pred_masks, pred_embeddings = self.model.evaluate(
#                 images_clip=image_clip,
#                 images=image_sam,
#                 input_ids=input_ids,
#                 resize_list=[image_sam.shape[2:]],  # H, W
#                 original_size_list=original_size_list,
#                 max_new_tokens=32,
#                 tokenizer=self.tokenizer,
#             )
#             # output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

#             # text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
#             # text_output = text_output.replace("\n", "").replace("  ", " ")

#             for i, pred_mask in enumerate(pred_masks):
#                 if pred_mask.shape[0] == 0:
#                     continue

#                 pred_mask = pred_mask.detach().cpu().numpy()[0]
#                 pred_mask = pred_mask > 0

#                 pred_mask = torch.from_numpy(pred_mask).long().to('cuda')

#             all_pred_masks.append(pred_mask)

#             if len(pred_embeddings) == 0:
#                 continue
#             else:
#                 # Concatenate all embeddings for this sample
#                 pred_embedding = torch.cat(pred_embeddings, dim=0)  # Shape: [num_embeddings, embedding_dim]

#             # Ensure pred_embedding has shape [1, embedding_dim]
#             if pred_embedding.dim() == 1:
#                 pred_embedding = pred_embedding.unsqueeze(0)

#             # Add to the list of all embeddings
#             all_pred_embeddings.append(pred_embedding)

#             image_feature = self.model.get_visual_embs(image_sam)
#             all_image_features.append(image_feature)

#         all_pred_masks = torch.stack(all_pred_masks, dim=0)  # [batch_size, W, H]
#         all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)  # [batch_size, embedding_dim]
#         all_image_features = torch.cat(all_image_features, dim=0)  # 形状：[batch_size, C, H', W']
#         # Return a dictionary containing all results
#         return {
#             'image_pred': all_pred_masks,
#             'image_features': all_image_features,
#             'pred_embeddings': all_pred_embeddings,
#         }
    
# def make_model(config):
#     """
#     Build points and image models according to what is in the config
#     """
#     model_fusion = fusionNet(config)
#     # model_lisa = initialize_lisa_model(config)

#     # Build point cloud model
#     # if config['dataset'] == "scannetpp":
#     if config['dataset'] in ["scannet", "scannetpp"]:
#         model_points = MinkUNet(3, config["model_n_out"], config)
#     else:
#         model_points = MinkUNet(1, config["model_n_out"], config)

#     # Build image model
#     if config["images_encoder"] == "lisa":
#         model_images = initialize_lisa_model(config)
#     elif config["images_encoder"].find("vit_") != -1:
#         model_images = DinoVitFeatureExtractor(config, preprocessing=Preprocessing())
#     elif config["images_encoder"] == "maskclip":
#         model_images = maskClipFeatureExtractor(config, preprocessing=Preprocessing())
#     elif config["decoder"] == "dilation":
#         model_images = DilationFeatureExtractor(config, preprocessing=Preprocessing())
#     elif config["decoder"] == "ppkt":
#         model_images = PPKTFeatureExtractor(config, preprocessing=Preprocessing())
#     else:
#         raise Exception(f"Model not found: {config['decoder']}")

#     return model_points, model_images, model_fusion

# def initialize_lisa_model(config):
#     tokenizer = AutoTokenizer.from_pretrained(
#         config["version"],
#         cache_dir=None,
#         model_max_length=config["model_max_length"],
#         padding_side="right",
#         use_fast=False,
#     )
#     tokenizer.pad_token = tokenizer.unk_token
#     config["seg_token_idx"] = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

#     torch_dtype = torch.float32
#     if config["precision"] == "bf16":
#         torch_dtype = torch.bfloat16
#     elif config["precision"] == "fp16":
#         torch_dtype = torch.half

#     kwargs = {"torch_dtype": torch_dtype}
#     if config.get("load_in_4bit", False):
#         kwargs.update(
#             {
#                 "torch_dtype": torch.half,
#                 "load_in_4bit": True,
#                 "quantization_config": BitsAndBytesConfig(
#                     load_in_4bit=True,
#                     bnb_4bit_compute_dtype=torch.float16,
#                     bnb_4bit_use_double_quant=True,
#                     bnb_4bit_quant_type="nf4",
#                     llm_int8_skip_modules=["visual_model"],
#                 ),
#             }
#         )
#     elif config.get("load_in_8bit", False):
#         kwargs.update(
#             {
#                 "torch_dtype": torch.half,
#                 "quantization_config": BitsAndBytesConfig(
#                     llm_int8_skip_modules=["visual_model"],
#                     load_in_8bit=True,
#                 ),
#             }
#         )

#     model = LISAForCausalLM.from_pretrained(
#         config["version"],
#         low_cpu_mem_usage=True,
#         vision_tower=config["vision_tower"],
#         seg_token_idx=config["seg_token_idx"],
#         **kwargs
#     )

#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.bos_token_id = tokenizer.bos_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id

#     model.get_model().initialize_vision_modules(model.get_model().config)
#     vision_tower = model.get_model().get_vision_tower()
#     vision_tower.to(dtype=torch_dtype)

#     if config["precision"] == "bf16":
#         model = model.bfloat16().cuda()
#     elif config["precision"] == "fp16" and (not config.get("load_in_4bit", False)) and (not config.get("load_in_8bit", False)):
#         vision_tower = model.get_model().get_vision_tower()
#         model.model.vision_tower = None
#         import deepspeed

#         model_engine = deepspeed.init_inference(
#             model=model,
#             dtype=torch.half,
#             replace_with_kernel_inject=True,
#             replace_method="auto",
#         )
#         model = model_engine.module
#         model.model.vision_tower = vision_tower.half().cuda()
#     elif config["precision"] == "fp32":
#         model = model.float().cuda()

#     vision_tower = model.get_model().get_vision_tower()
#     vision_tower.to(device=config["local_rank"])

#     clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
#     transform = ResizeLongestSide(config["image_size"])

#     model.eval()

#     model_lisa = ModelImagesWrapper(
#         model=model,
#         tokenizer=tokenizer,
#         clip_image_processor=clip_image_processor,
#         transform=transform,
#         config=config
#     )

#     return model_lisa