_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained',
                      # checkpoint='/cfs/cfs-31b43a0b8/personal/mingzhenzhu/DenseCLIP/detection/pretrained/kdep_D_imagenet_T_msr50_S_r18_ep90.pth'),
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_imagenet_T_clipr50_S_r18.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_T_in1k_T_clipr50_S_r18_200ep_tv.pth',
                      checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_L_lvis_T_clipr50_S_r18_ep90.pth',
                      )),
    neck=dict(
        in_channels=[64, 128, 256, 512]),
    roi_head=dict(
        bbox_head=dict(num_classes=1203), mask_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(samples_per_gpu=2,
            train=dict(dataset=dict(pipeline=train_pipeline)))

# optimizer settings
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    )
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=0.1, norm_type=2))

# evaluate settings
evaluation = dict(
    interval=12,
    metric=['bbox', 'segm'])


