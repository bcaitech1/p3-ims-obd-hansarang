checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    # interval=50,
    # hooks=[
    #     dict(type='TextLoggerHook'),
    #     # dict(type='TensorboardLoggerHook')
    # ]
    interval=200,

    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='Default',
                 name='Default'
             ))
    ]
)
log_config2 = dict(

)
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
