_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='SwinTransformer'),
#         conv_cfg=dict(type='ConvAWS'),
#         output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
#             rfp_inplanes=256,
            type='SwinTransformer',
#             depth=50,
#             num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
#             norm_cfg=dict(type='BN', requires_grad=True),
#             norm_eval=True,
#             conv_cfg=dict(type='ConvAWS'),
#             pretrained='/opt/ml/code/mmdetection_trash/work_dirs/cascade_swin/swin_small_patch4_window7_224.pth',
#             style='pytorch')))
        )))