# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('nvidia-smi')


# %%
import os, time
import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor) # 앞의 2개 --> mmdet/datasets/builder.py에 존재 / 마지막 1개 --> 
from mmdet.models import build_detector #mmdet/models/builder.py에 존재
from mmdet.apis import (train_detector, single_gpu_test, init_detector, inference_detector, show_result_pyplot) 
    # train_detector --> mmdet/apis/train.py에 존재, init_detector, inference_detector, show_result_pyplot --> mmdet/apis/inference.py에 존재, single_gpu_test -->  mmdet/apis/test.py에 존재
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import json

# %% [markdown]
# ## train part

# %%
#temp
'''
config_dir_name = 'detectors'
config_file_name = 'detectors_htc_r50_2x_coco'
cfg = Config.fromfile('../configs/'+config_dir_name+'/'+config_file_name+'.py')
working_dir_name = '../work_dirs/' + config_file_name +'_'+str(cfg.runner.max_epochs)

try:
    if not os.path.exists(working_dir_name):
        os.makedirs(working_dir_name)
except OSError :
    print('Error: Creating directory :: ' + working_dir_name)
cfg_str = str(cfg)[str(cfg).find(')')+3:]
cfg_dict = eval(cfg_str); 
with open(working_dir_name+'/config.json','w') as config_json_file :
    json.dump(cfg_dict, config_json_file, indent="\t")
#config 저장 확인
with open(working_dir_name+'/config.json', 'r') as f :
    json_data = json.load(f)
json_data
'''


# %%
classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
#cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
#파일 교체 시 여기 이름 바꾸기
config_dir_name = 'detectors'
config_file_name = 'DetectoRS_mstrain_400_1200_r50'
cfg = Config.fromfile('../configs/'+config_dir_name+'/'+config_file_name+'.py')

PREFIX = '/opt/ml/input/data/'

#data 경로 설정
cfg.data_root = PREFIX

# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'
cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

cfg.data.samples_per_gpu = 16    #batch size
cfg.data.workers_per_gpu = 4     #worker num

cfg.seed=42 #42로 고정
cfg.gpu_ids = [0]

#epoch 수 조정
cfg.runner.max_epochs = 24

#class 개수 설정
cfg.model.bbox_head[0].num_classes = 11 
cfg.model.bbox_head[1].num_classes = 11
cfg.model.bbox_head[2].num_classes = 11

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

#working directory 이름 설정
working_dir_name = '../work_dirs/' + config_file_name +'_'+str(cfg.runner.max_epochs)
cfg.work_dir = working_dir_name

# %% [markdown]
# ## wandb setting

# %%
group_name = 'detectors'; project_name = 'gjtak'; run_name = config_file_name
config_list = {
    'epoch' : cfg.runner.max_epochs,
    'batch_size' :  cfg.data.samples_per_gpu,
    'optimizer' : cfg.optimizer,
    'optimizer_config' : cfg.optimizer_config,
    'lr_config' : cfg.lr_config,
}
cfg.log_config.hooks[1].init_kwargs['group']=group_name # group name(option)
cfg.log_config.hooks[1].init_kwargs['name'] = run_name # run name
cfg.log_config.hooks[1].init_kwargs['config'] = config_list # config

# %% [markdown]
# ### config 저장

# %%
# config 객체 -> 파일(json)로 저장(저장 장소 = working directory와 같이)
#working directory 없으면 생성하기
try:
    if not os.path.exists(working_dir_name):
        os.makedirs(working_dir_name)
except OSError :
    print('Error: Creating directory :: ' + working_dir_name)

#cfg 객체 -> string(str이용) -> dict(eval이용) -> json 으로 저장
cfg_str = str(cfg)[str(cfg).find(')')+3:]
cfg_dict = eval(cfg_str)
with open(working_dir_name+'/config.json','w') as config_json_file :
    json.dump(cfg_dict, config_json_file, indent="\t")


# %%
#config 저장 확인
with open(working_dir_name+'/config.json', 'r') as f :
    json_data = json.load(f)
#json_data


# %%
model = build_detector(cfg.model)


# %%
datasets = [build_dataset(cfg.data.train)]


# %%
start = time.time()  # 시작 시간 저장
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
print("time(sec) :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간(단위 : 초)

# %% [markdown]
# # Inference part

# %%
# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')

# %% [markdown]
# ## \[option\] box inference image result

# %%
model_for_img_test = init_detector(cfg, checkpoint_path, device='cuda:0')


# %%
img_dir_num = ['01', '02', '03'] 
img_num = '0003'
img = '/opt/ml/input/data/batch_'+img_dir_num[1]+'_vt/'+img_num+'.jpg'
result = inference_detector(model_for_img_test, img)
# show the results
show_result_pyplot(model_for_img_test, img, result)

# %% [markdown]
# # Real Inference

# %%
epoch = cfg.runner.max_epochs
cfg.model.train_cfg = None


# %%
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


# %%
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


# %%
output = single_gpu_test(model, data_loader, show_score_thr=0.05)


# %%
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
imag_ids = coco.getImgIds()

class_num = 11
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
        
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names

submission.to_csv(os.path.join(cfg.work_dir, 'submission_'+config_file_name+f'_{epoch}.csv'), index=None)
submission.head()


# %%



