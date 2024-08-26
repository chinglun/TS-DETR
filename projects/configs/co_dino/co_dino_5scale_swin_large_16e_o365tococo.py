_base_ = [
    'MutiAugment98.py'
]
pretrained = 'models/swin_large_patch4_window12_384_22k.pth'
load_from = 'models/co_dino_5scale_swin_large_22e_o365.pth'
#pretrained = None
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        pretrained=pretrained),
    neck=dict(in_channels=[192, 192*2, 192*4, 192*8]),
    query_head=dict(
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=50)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)