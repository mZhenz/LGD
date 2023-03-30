_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    # pretrained='/cfs/cfs-31b43a0b8/personal/mingzhenzhu/DenseCLIP/detection/pretrained/kdep_D_imagenet_T_msr50_S_r18_ep90.pth',
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained',
                      # checkpoint='torchvision://resnet50',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_imagenet_T_clipr50_S_r18.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_imagenet_T_clipr50_S_r18_v2.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_T_in1k_T_clipr50_S_r18_200ep_tv.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_T_in1k_T_clipr50_S_r18_90ep_tv.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_coco_L_coco_T_clipr50_S_r18_900ep_tv.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_L_coco_T_clipr50_S_r18_ep90_tv.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_coco_L_coco_T_clipr50_S_r18_90ep_new.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_L_coco_T_clipr50_S_r18_ep90_new.pth',
                      # checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_L_coco_T_clipr50_S_r18_ep90_lr0.03.pth',
                      checkpoint='/apdcephfs/share_1227775/mingzhenzhu/ClipDistiller/pretrained/clipdistv9_D_in1k_L_coco_T_clipr50_S_r18_ep90_1118.pth',
                      )),
    neck=dict(in_channels=[64, 128, 256, 512]),
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    )
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=0.1, norm_type=2))
