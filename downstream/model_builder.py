import torch
from model import MinkUNet, SPVCNN
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def load_state_with_same_shape(model, weights):
    """
    Load common weights in two similar models
    (for instance between a pretraining and a downstream training)
    """
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith("model."):
        weights = {k.partition("model.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("model_points."):
        weights = {k.partition("model_points.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("module."):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition("module.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("encoder."):
        print("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v
        for k, v in weights.items()
        if (k in model_state and v.size() == model_state[k].size())
    }
    removed_weights = {
        k: v
        for k, v in weights.items()
        if not (k in model_state and v.size() == model_state[k].size())
    }
    print("Loading weights:" + ", ".join(filtered_weights.keys()))
    print("")
    print("Not loading weights:" + ", ".join(removed_weights.keys()))
    return filtered_weights


def make_model(config, load_path=None):
    """
    Build the points model according to what is in the config
    """

    assert not config[
        "normalize_features"
    ], "You shouldn't normalize features for the downstream task"
    # model = MinkUNet(1, config["model_n_out"], config)
    # model = SPVCNN(1, config["model_n_out"], config)
    # model = MinkUNet(3, config["model_n_out"], config)
    print("MinkUNet:", MinkUNet)

    if config['dataset'] in ["scannet", "scannetpp"]:
        model = MinkUNet(3, config["model_n_out"], config)
    else:
        # model_points = SPVCNN(1, config["model_n_out"], config)
        model = MinkUNet(1, config["model_n_out"], config)
    if load_path:
        print("Training with pretrained model")
        checkpoint = torch.load(load_path, map_location="cpu")
        if "config" in checkpoint:
            for cfg in ("voxel_size", "cylindrical_coordinates"):
                assert checkpoint["config"][cfg] == config[cfg], (
                    f"{cfg} is not consistant. "
                    f"Checkpoint: {checkpoint['config'][cfg]}, "
                    f"Config: {config[cfg]}."
                )
        if set(checkpoint.keys()) == set(["epoch", "model", "optimizer", "train_criterion"]):
            print("Pre-trained weights are coming from DepthContrast.")
            pretraining_epochs = checkpoint["epoch"]
            print(f"==> Number of pre-training epochs {pretraining_epochs}")
            checkpoint = checkpoint["model"]
            if list(checkpoint.keys())[0].startswith("module."):
                print("Loading multigpu weights with module. prefix...")
                checkpoint = {k.partition("module.")[2]: checkpoint[k] for k in checkpoint.keys()}
            voxel_net_suffix = "trunk.2."
            checkpoint = {
                key.partition(voxel_net_suffix)[2]: checkpoint[key]
                for key in checkpoint.keys() if key.startswith(voxel_net_suffix)
            }
            print(f"==> Number of loaded weight blobs {len(checkpoint)}")
            checkpoint = {"model_points": checkpoint}
        key = "model_points" if "model_points" in checkpoint else "state_dict"
        filtered_weights = load_state_with_same_shape(model, checkpoint[key])
        model_dict = model.state_dict()
        model_dict.update(filtered_weights)
        model.load_state_dict(model_dict)
    if config["freeze_layers"]:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    return model


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

    model_images = ModelImagesWrapper(
        model=model,
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor,
        transform=transform,
        config=config
    )

    return model_images

def preprocess(
        x, 
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


class ModelImagesWrapper(nn.Module):
    """
    A wrapper class for LISAForCausalLM to freeze and use it for inference only.
    """
    def __init__(self, model, tokenizer, clip_image_processor, transform, config, 
                 checkpoint_path='/home/jiaxin.huang/jiaxin/reasonseg/tensorboard/tensorboard_logs/version_23/checkpoints/epoch=59-step=6000.ckpt'):
        super(ModelImagesWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.transform = transform
        self.config = config
        self.model.eval()  # Freeze LISA model for inference
        self.embedding_linear = nn.Linear(256, 512)
        self.embedding_linear.to(device=next(self.model.parameters()).device)
        if checkpoint_path:
            self.load_linear_weights(checkpoint_path)

    def load_linear_weights(self, checkpoint_path):
        """
        Load weights for embedding_linear layer from the saved checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']  # Assuming the state_dict is under 'state_dict' key
        
        # Load weights and bias for embedding_linear
        if 'embedding_linear.weight' in state_dict:
            self.embedding_linear.weight.data.copy_(state_dict['embedding_linear.weight'])
        if 'embedding_linear.bias' in state_dict:
            self.embedding_linear.bias.data.copy_(state_dict['embedding_linear.bias'])

    def forward(self, batch):
        """
        Forward function that performs inference using LISA model.
        Args:
            batch: Dictionary containing:
                - image
                - description
        Returns:
            Dictionary containing:
                - pred_embeddings: Extracted embeddings for [SEG] tokens
                - image_pred: Predicted mask for the image
        """
        device = next(self.model.parameters()).device

        image_np, prompt = batch  # Unpack a single image and description
        # Process prompt for input
        conv_type = self.config.get("conv_type", "llava_v1") 
        conv = conversation_lib.conv_templates[conv_type].copy()
        conv.messages = []

        prompt = str(prompt)
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + "Please output segmentation mask."
        if self.config["use_mm_start_end"]:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # Prepare image data for model input
        original_size_list = [image_np.shape[:2]]  # Extract original H, W

        # Process image for CLIP-based input (image_clip)
        image_clip = (
            self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0)
            .to(device)
        )

        # Adjust precision if needed
        if self.config["precision"] == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.config["precision"] == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # Transform and preprocess the image for SAM input
        image_sam = (
            preprocess(torch.from_numpy(image_np).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(device)
        )

        # Adjust precision if needed
        if self.config["precision"] == "bf16":
            image_sam = image_sam.bfloat16()
        elif self.config["precision"] == "fp16":
            image_sam = image_sam.half()
        else:
            image_sam = image_sam.float()

        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)

        # Perform inference using model's evaluate() function
        output_ids, pred_masks, pred_embeddings = self.model.evaluate(
            images_clip=image_clip,
            images=image_sam,
            input_ids=input_ids,
            resize_list=[image_sam.shape[2:]],  # H, W
            original_size_list=original_size_list,
            max_new_tokens=32,
            tokenizer=self.tokenizer,
        )

        # Decode text output (optional, for debugging or analysis)
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            pred_mask = torch.from_numpy(pred_mask).long().to('cuda')

        # all_pred_masks.append(pred_mask)

        if len(pred_embeddings) == 0:
            # If no embeddings are generated, return empty results
            return {
                'pred_embeddings': None,
                'image_pred': None
            }

        # Concatenate embeddings if multiple are present
        pred_embedding = torch.cat(pred_embeddings, dim=0)  # Shape: [num_embeddings, embedding_dim]
        if pred_embedding.dim() == 1:
            pred_embedding = pred_embedding.unsqueeze(0)

        # Pass embeddings through linear layer
        pred_embedding = self.embedding_linear(pred_embedding.float())

        # Return results as a dictionary
        return {
            'pred_embeddings': pred_embedding,  # Predicted embeddings for the description
            'image_pred': pred_mask  # Predicted mask for the input image
        }


    # def forward(self, batch):
    #     """
    #     Forward function that performs inference using LISA model.
    #     Args:
    #         batch: Dictionary containing:
    #             - image
    #             - description
    #     Returns:
    #         Dictionary containing:
    #             - text_embeddings: Extracted embeddings for [SEG] tokens
    #     """
    #     device = next(self.model.parameters()).device

    #     all_embeddings = []
    #     all_pred_masks = []
    #     for image_np, prompt in batch:
    #         single_sample_embeddings = []  # 用于存储当前样本的所有嵌入
    #         # image_np = image_np.numpy() # [W, H, C]
                
    #         conv_type = self.config.get("conv_type", "llava_v1") 
    #         conv = conversation_lib.conv_templates[conv_type].copy()
    #         conv.messages = []
                
    #         prompt = str(prompt)
    #         prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + "Please output segmentation mask."
    #             #prompt = DEFAULT_IMAGE_TOKEN + "\n" + " ".join(prompt)
    #         if self.config["use_mm_start_end"]:
    #             replace_token = (
    #                 DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    #             )
    #             prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                
    #         conv.append_message(conv.roles[0], prompt)
    #         conv.append_message(conv.roles[1], "")
    #         prompt = conv.get_prompt()

    #             # Prepare image data for model input
    #         original_size_list = [image_np.shape[:2]]  # Extract original H, W

    #             # Process image for CLIP-based input (image_clip)
    #         image_clip = (
    #             self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
    #             .unsqueeze(0)
    #             .to(device)
    #         )

    #         # Adjust precision if needed
    #         if self.config["precision"] == "bf16":
    #             image_clip = image_clip.bfloat16()
    #         elif self.config["precision"] == "fp16":
    #             image_clip = image_clip.half()
    #         else:
    #             image_clip = image_clip.float()

    #             # Transform and preprocess the image for SAM input
    #         image_sam = (
    #             preprocess(torch.from_numpy(image_np).permute(2, 0, 1).contiguous())
    #             .unsqueeze(0)
    #             .to(device)
    #         )
    #             #print(f"image_sam shape: {image_sam.shape}")

    #             # Adjust precision if needed
    #         if self.config["precision"] == "bf16":
    #             image_sam = image_sam.bfloat16()
    #         elif self.config["precision"] == "fp16":
    #             image_sam = image_sam.half()
    #         else:
    #             image_sam = image_sam.float()

    #             # Tokenize prompt
    #         input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
    #         input_ids = input_ids.unsqueeze(0).to(device)

    #             # Perform inference using model's evaluate() function
    #         output_ids, pred_masks, pred_embeddings = self.model.evaluate(
    #             images_clip=image_clip,
    #             images=image_sam,
    #             input_ids=input_ids,
    #             resize_list=[image_sam.shape[2:]],  # H, W
    #             original_size_list=original_size_list,
    #             max_new_tokens=32,
    #             tokenizer=self.tokenizer,
    #         )
    #         # print(f"pred_embeddings: {pred_embeddings}")
    #         output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    #         text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
    #         text_output = text_output.replace("\n", "").replace("  ", " ")
    #         # print("text_output: ", text_output)

    #         for i, pred_mask in enumerate(pred_masks):
    #             if pred_mask.shape[0] == 0:
    #                 continue

    #             pred_mask = pred_mask.detach().cpu().numpy()[0]
    #             pred_mask = pred_mask > 0

    #             pred_mask = torch.from_numpy(pred_mask).long().to('cuda')

    #         all_pred_masks.append(pred_mask)

    #         if len(pred_embeddings) == 0:
    #             continue
    #         else:
    #             # Concatenate all embeddings for this sample
    #             pred_embedding = torch.cat(pred_embeddings, dim=0)  # Shape: [num_embeddings, embedding_dim]

    #         # Ensure pred_embedding has shape [1, embedding_dim]
    #         if pred_embedding.dim() == 1:
    #             pred_embedding = pred_embedding.unsqueeze(0)

    #         # Add to the list of all embeddings
    #         all_pred_embeddings.append(pred_embedding)

    #         image_feature = self.model.get_visual_embs(image_sam)
    #         # all_image_features.append(image_feature)

    #     all_pred_masks = torch.stack(all_pred_masks, dim=0)  # [batch_size, W, H]
    #     all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)  # [batch_size, embedding_dim]
    #     # all_image_features = torch.cat(all_image_features, dim=0)  # 形状：[batch_size, C, H', W']
    #     # Return a dictionary containing all results
    #     return {
    #         'image_pred': all_pred_masks,
    #         # 'image_features': all_image_features,
    #         'pred_embeddings': all_pred_embeddings,
    #     }

