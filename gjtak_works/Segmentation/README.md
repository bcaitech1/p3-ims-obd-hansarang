# P-stage3 Segmentation 한사랑개발회 탁금지 실험결과 정리

|Date|Model Name|Encoder|Arguments|WanDB Link|LB score|ETC|
|----|----------|-------|---------|----------|--------|---|
|2021-04-22|Unet|resnet50|loss function=CrossEntropyLoss  optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)  pretrained=imagenet<br>batch size=10<br>epoch=10|None||efficientnet 계열보다 빠른 학습 시간을 보임|
|2021-04-22|Unet|resnext50_32x4d|loss function=CrossEntropyLoss  optimizer=adam(params=model.parameters() / learning rate=1e-4 / weight decay=1e-6)  pretrained=imagenet  batch size=10  epoch=10|<https://wandb.ai/pstage12/gjtak/runs/d2ts77ua?workspace=user-atica>|efficientnet 계열보다 빠른 학습 시간을 보임  validataion 상으로는 지금까지 진행한 테스트 결과 중 가장 좋은 결과를 보임|
|테스트1|테스트2|테스트3|
