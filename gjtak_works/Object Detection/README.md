# P-stage3 Object Detection 한사랑개발회 탁금지 실험일지
## 실험 일지
<details>
    <summary>2021-05 3주차(Object Detection 1주차)</summary>
        - 05-10-Mon<br>
        <p>&ensp - mmdetection 기본 baseline code 실행(faster rcnn-resnet50)</p>
        - 05-11-Tue<br>
        <p>&ensp - git branch 생성(gjtak_branch)</p>
</details>
<details>
    <summary>2021-05 4주차(Object Detection 2주차)</summary>
        - 
</details>

## 실험 결과 정리

|Date|Model Name|Encoder|Arguments|WanDB Link|LB score|ETC|
|----|----------|-------|---------|----------|--------|---|
|2021-05-11|Unet|resnet50|loss function=CrossEntropyLoss<br>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br>pretrained=imagenet<br>batch size=10<br>epoch=10|[resnet50](https://wandb.ai/pstage12/gjtak/runs/3grmk6yo?workspace=user-atica)|efficientnet 계열보다 빠른 학습 시간을 보임|
|2021-05-11|Unet|resnet50|loss function=CrossEntropyLoss<br>optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)<br>pretrained=imagenet<br>batch size=10<br>epoch=10|[resnet50](https://wandb.ai/pstage12/gjtak/runs/3grmk6yo?workspace=user-atica)|efficientnet 계열보다 빠른 학습 시간을 보임|
