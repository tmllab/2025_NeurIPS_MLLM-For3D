_base_ = './maskclip_plus_r50_deeplabv2_r101-d8_512x512_2k_voc12aug_20.py'

model = dict(
    decode_head=dict(
        clip_channels=3072,
        channels=768,
        text_channels=768,
        text_embeddings_path='pretrain/voc_RN50x16_clip_text.pth',
        clip_weights_path='pretrain/RN50x16_clip_weights.pth',
        clip_cfg=dict(
            type='ResNetClip',
            stem_channels=96,
            base_channels=96,
            depth='50x16'
        ),
    )
)