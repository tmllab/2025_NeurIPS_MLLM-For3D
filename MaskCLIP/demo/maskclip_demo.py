from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import torch
from tools.maskclip_utils.prompt_engineering import zeroshot_classifier, prompt_templates

config_file = '../configs/maskclip/maskclip_vit16_512x512_demo.py'
config = mmcv.Config.fromfile(config_file)
checkpoint_file = '../pretrain/ViT16_clip_backbone.pth'

img = 'demo.png'
fg_classes = ['pedestrian', 'car', 'bicycle']
bg_classes = ['road', 'building']

# text_embeddings = zeroshot_classifier('ViT-B/16', fg_classes+bg_classes, prompt_templates)
# text_embeddings = text_embeddings.permute(1, 0).float()
# print(text_embeddings.shape)
# torch.save(text_embeddings, '../pretrain/demo_ViT16_clip_text.pth')

num_classes = len(fg_classes + bg_classes)
config.model.decode_head.num_classes = num_classes
config.model.decode_head.text_categories = num_classes

config.data.test.fg_classes = fg_classes
config.data.test.bg_classes = bg_classes

# config.model.decode_head.num_vote = 1
# config.model.decode_head.vote_thresh = 1.
# config.model.decode_head.cls_thresh = 0.5

# build the model from a config file and a checkpoint file
model = init_segmentor(config, checkpoint_file, device='cuda:0')

# test a single image
result = inference_segmentor(model, img)

# show the results
show_result_pyplot(model, img, result, None)

