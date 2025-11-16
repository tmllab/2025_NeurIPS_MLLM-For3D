_base_ = './maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.py'

model = dict(
    decode_head=dict(
        ks_thresh=1.0,
        pd_thresh=0.5,
        # conf_thresh = 0.1,
    ),
)