_base_ = './maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py'

model = dict(
    decode_head=dict(
        distill=True,
        distill_labeled=True,
        distill_weight=1
    ),
)