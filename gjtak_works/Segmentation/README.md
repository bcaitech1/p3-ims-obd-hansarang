# P-stage3 Segmentation 한사랑개발회 탁금지 실험결과 정리

### Unet

| Date       | Model Name | Encoder              | Arguments                                                    | WanDB Link                                                   | mIoU(latest) | LB score      |
| ---------- | ---------- | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | ------------- |
| 2021-04-29 | Unet       | resnet50             | loss function=CrossEntropyLoss<br>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br>pretrained=imagenet<br>batch size=10<br>epoch=10 | [resnet50](https://wandb.ai/pstage12/gjtak/runs/3grmk6yo?workspace=user-atica) | 0.3734       | no submission |
| 2021-04-29 | Unet       | resnext50_32x4d      | loss function=CrossEntropyLoss<br>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br>pretrained=imagenet<br>batch size=10<br>epoch=10 | [resnext50_32x4d](https://wandb.ai/pstage12/gjtak/runs/d2ts77ua?workspace=user-atica) | 0.3907       | no submission |
| 2021-04-29 | Unet       | resnet101            | loss function=CrossEntropyLoss<br>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br>pretrained=imagenet<br>batch size=10<br>epoch=10 | [resnet101](https://wandb.ai/pstage12/gjtak/runs/1zmluwoa?workspace=user-atica) | 0.35         | no submission |
| 2021-04-30 | Unet       | resnext101_32x8d     | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch size=8<br/>epoch=10 | [resnet101_32x8d](https://wandb.ai/pstage12/gjtak/runs/28ytanf5?workspace=user-atica) | 0.3638       | no submission |
| 2021-04-30 | Unet       | efficientnet-b5      | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch size=8<br/>epoch=10 | [efficientnet-b5](https://wandb.ai/pstage12/gjtak/runs/1bodr5zi?workspace=user-atica) |              | no submission |
| 2021-04-30 | Unet       | efficientnet-b7      | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch_size(train/test) = 4/5<br/>epoch=10 | [efficientnet-b7](https://wandb.ai/pstage12/gjtak/runs/3natzzsf?workspace=user-atica) |              | no submission |
| 2021-04-30 | Unet       | timm-efficientnet-b4 | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch_size(train/test) = 4<br/>epoch=10 | [timm-efficientnet-b4](https://wandb.ai/pstage12/gjtak/runs/286gclty?workspace=user-atica) |              |               |
| 2021-05-01 | Unet       | densenet121          | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch_size(train/test) = 8/5<br/>epoch=10 | [densenet121](https://wandb.ai/pstage12/gjtak/runs/1ebc9hal?workspace=user-atica) |              |               |
| 2021-05-02 | Unet       | efficientnet-b0      | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch size=8<br/>epoch=10 | [efficientnet-b0](https://wandb.ai/pstage12/gjtak/runs/4y8gh6mp?workspace=user-atica) |              |               |
| 2021-05-02 | Unet       | efficientnet-b4      | loss function=CrossEntropyLoss<br/>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br/>pretrained=imagenet<br/>batch size=8<br/>epoch=10 | [efficientnet-b4](https://wandb.ai/pstage12/gjtak/runs/39jdumpx?workspace=user-atica) |              |               |
|            |            |                      |                                                              |                                                              |              |               |



### Unet++

| Date       | Model Name | Encoder         | Arguments | WanDB Link                                                   | mIoU(latest) | LB score |
| ---------- | ---------- | --------------- | --------- | ------------------------------------------------------------ | ------------ | -------- |
| 2021-04-30 | Unet++     | resnet50        |           | [resnet50](https://wandb.ai/pstage12/gjtak/runs/x42kojq1?workspace=user-atica) |              |          |
| 2021-04-30 | Unet++     | resnext50_32x4d |           | [resnext50_32x4d](https://wandb.ai/pstage12/gjtak/runs/3w73pyxa?workspace=user-atica) |              |          |
| 2021-04-30 | Unet++     | efficientnet-b4 |           | [efficientnet-b4](https://wandb.ai/pstage12/gjtak/runs/1lfc58d2?workspace=user-atica) |              |          |
| 2021-05-01 | Unet++     | efficientnet-b7 |           | [efficientnet-b7](https://wandb.ai/pstage12/gjtak/runs/26rujadx?workspace=user-atica) |              |          |
| 2021-05-02 | Unet++     | efficientnet-b0 |           | [efficientnet-b0](https://wandb.ai/pstage12/gjtak/runs/3lptkh2t?workspace=user-atica) |              |          |



### DeepLabV3

| Date       | Model Name | Encoder         | Arguments                                                    | WanDB Link                                                   | mIoU(latest) | LB score      |
| ---------- | ---------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | ------------- |
| 2021-05-03 | DeepLabV3  | efficientnet-b4 | batch size(train/test) = 4/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = X<br/>criterion = CrossEntropy()<br/>epoch=10 | [efficientnet-b4](https://wandb.ai/pstage12/gjtak/runs/heje5fhr?workspace=user-atica) | 0.4243       | no submission |
| 2021-05-08 | DeepLabV3  | efficientnet-b4 | [k fold1]<br/>batch size(train/test) = 3/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = CosineAnnaealingLR(T_max = 50, eta_min= 0)<br/>criterion = CrossEntropy()<br />epoch=5 | [kfold1](https://wandb.ai/pstage12/gjtak/runs/3rbiidl3?workspace=user-atica) | 0.4249       | no submission |
| 2021-05-08 | DeepLabV3  | efficientnet-b4 | [k fold2]<br/>batch size(train/test) = 3/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = CosineAnnaealingLR(T_max = 50, eta_min= 0)<br/>criterion = CrossEntropy()<br />epoch=5 | [kfold2](https://wandb.ai/pstage12/gjtak/runs/7sso4jrd?workspace=user-atica) | 0.4231       | no submission |
| 2021-05-08 | DeepLabV3  | efficientnet-b4 | [k fold3]<br/>batch size(train/test) = 3/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = CosineAnnaealingLR(T_max = 50, eta_min= 0)<br/>criterion = CrossEntropy()<br />epoch=5 | [kfold3](https://wandb.ai/pstage12/gjtak/runs/2p64slhb?workspace=user-atica) | 0.4425       | no submission |
| 2021-05-08 | DeepLabV3  | efficientnet-b4 | [k fold4]<br/>batch size(train/test) = 3/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = CosineAnnaealingLR(T_max = 50, eta_min= 0)<br/>criterion = CrossEntropy()<br />epoch=5 | [kfold4](https://wandb.ai/pstage12/gjtak/runs/1fiqltll?workspace=user-atica) | 0.4317       | no submission |
| 2021-05-08 | DeepLabV3  | efficientnet-b4 | [k fold5]<br/>batch size(train/test) = 3/5<br/>pretrained = imagenet<br/>optimizer = adam(learning rate = 1e-4, weight decay = 1e-6)<br/>scheduler = CosineAnnaealingLR(T_max = 50, eta_min= 0)<br/>criterion = CrossEntropy()<br />epoch=5 | [kfold5](https://wandb.ai/pstage12/gjtak/runs/3s09ekad?workspace=user-atica) | 0.4403       | no submission |

