checkpoint_config = dict(interval=1)
# yapf:disable

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # wandb
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='kw_ojd_p3_2',
                #name = 'Your EXP'#run name 
                )
        ),
    ])

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
