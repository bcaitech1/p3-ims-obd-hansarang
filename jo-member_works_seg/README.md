# 한사랑개발회 조원 T1200 실험 기록

## Stage 3-1 Segmentation



| Date       | Run Name                          | Model Name    | Arguments                                                    | WanDB Link                                                   | ETC                                                          |
| ---------- | --------------------------------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021-05-04 | Focal+Dice+b5+Cosinewarmup        | DeepLabV3Plus | augmentation='NormalizedAugmentation', batch_size=8, criterion='focal+dice', dataset='TrashDataset', encoder_name='timm-efficientnet-b5', encoder_weights='noisy-student', epochs=25, log_interval=20, lr=5e-6, lr_decay_step=1, model='DeepLabV3Plus', name='b5_ver', optimizer='AdamP', seed=42, valid_batch_size=8) | [link](https://wandb.ai/pstage12/chowon/runs/3qn89gjv/overview?workspace=user-jo_member) | Focal gamma = 0.5 optimizer  weight decay =1e-3 eps=1e-6     |
| 2021-05-04 | Focal+Dice+b5+Cosinewarmup+Steplr |               |                                                              |                                                              | 최고의 score을 가진 model을 불러다가 작은 lr + steplr로 적은수의 epoch을 적용해서 최적화 시킨다 |
|            |                                   |               |                                                              |                                                              |                                                              |
|            |                                   |               |                                                              |                                                              |                                                              |
|            |                                   |               |                                                              |                                                              |                                                              |

해봐야 될것들

Test time augmentation적용

Augemtataion 적용

- Rotate
- CLAHE
- Cutout

이 3가지 적용후 가장 좋은 model로 psudo labeling(적당한 확률값이) + validataion data도 학습set으로 껴서 최적만큼 실행

이렇게 해서 나온 b5 model과 resnet model을 앙상블 하여 결과를 일단 얻어냄