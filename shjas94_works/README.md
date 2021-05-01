# 한사랑개발회 허재섭 T1224 실험 기록

## Stage 3-1 Segmentation

| Run Name           | Model Name | Arguments(detail)                                                                                                                                                                                                              | WanDB Link                                                            |
| ------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| DeepLabV3_cosinelr | DeepLabV3  | batch_size=8, criterion='cross_entropy', <br>lr=5e-05, <br>model='DeepLabV3', num_epochs=10, optimizer='MADGRAD', run_name='DeepLabV3_cosinelr', scheduler='CosineAnnealingLR', seed=42, val_batch_size=8, weight_decay=0.0001 | https://wandb.ai/pstage12/jaesub/runs/3l1vdffh?workspace=user-shjas94 |
