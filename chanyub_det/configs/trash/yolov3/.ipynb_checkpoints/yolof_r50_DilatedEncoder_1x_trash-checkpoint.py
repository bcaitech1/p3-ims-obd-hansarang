_base_ = [
    '../../_base_/models/yolof_r50_DilatedEncoder_1x_trash.py',
    '../dataset.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=11
        )
    )
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
