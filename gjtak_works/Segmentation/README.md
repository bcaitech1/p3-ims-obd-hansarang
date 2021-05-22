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

| Date | Model Name | Encoder | Arguments | WanDB Link | mIoU(latest) | LB score |
| ---- | ---------- | ------- | --------- | ---------- | ------------ | -------- |
|      |            |         |           |            |              |          |
|      |            |         |           |            |              |          |
|      |            |         |           |            |              |          |

