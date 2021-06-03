"""Mission"""

# 기존 baseline code로 제공되었던 faster_rcnn architecture 그대로 가져오기
_base_ = [
    './faster_rcnn_r50_fpn_1x_trash.py'
]

model = dict(
    # model의 neck 부분만 PAFPN으로 변경
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
)
