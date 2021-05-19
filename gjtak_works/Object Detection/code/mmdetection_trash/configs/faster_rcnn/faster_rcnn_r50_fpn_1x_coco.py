_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_pipeline_fix.py',
    '../_base_/schedules/schedule_1x_adam.py',
    '../_base_/default_runtime.py'
]

#for cosine annealing
#'../_base_/schedules/schedule_1x_adam_cosine_annealing.py', 

#for SGD
#'../_base_/schedules/schedule_1x.py'

#for adam
#'../_base_/schedules/schedule_1x_adam.py'

#for adamW
#'../_base_/schedules/schedule_1x_adamW.py'