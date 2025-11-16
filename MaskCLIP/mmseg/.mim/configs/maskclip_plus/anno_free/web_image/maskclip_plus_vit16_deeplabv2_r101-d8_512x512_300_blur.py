_base_ = [
    '../../../_base_/models/maskclip_plus_vit16.py', '../../../_base_/datasets/web_image.py', 
    '../../../_base_/default_runtime.py', '../../../_base_/schedules/schedule_300.py'
]
img_dir = 'blur'
num_class = 20
suppress_labels = list(range(0, num_class))
model = dict(
    decode_head=dict(
        num_classes=num_class,
        text_categories=num_class,
        text_embeddings_path=f'pretrain/{img_dir}_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
    ),
)

find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', suppress_labels=suppress_labels),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=4,
    train=dict(img_dir=img_dir, pipeline=train_pipeline, data_name=img_dir),
    test=dict(img_dir=img_dir, data_name=img_dir)
    # test=dict(split=f'{img_dir}.txt', data_name=img_dir)
)