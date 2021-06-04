# 이미지 분류
> Private team mean_ap: 0.4714
![image](https://user-images.githubusercontent.com/54899906/120759377-ec260380-c54d-11eb-9768-450f646adec8.png)
---
## 대회 개요
사진 속 쓰레기의 종류를 분류하는 object detection task
- 데이터
  1. 전체 이미지 개수 : 4109장
  2. 11 class : UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  3. 이미지 크기 : (512, 512)
- 평가 방법
  - Test set의 mAP50
---
## 앙상블 전 내 모델
- model : Cascade_mask_RCNN
- backbone : Swin_Transformer_Small
- neck : PAFPN
---
## 시도해본 것들
- SwinTransformer를 mmdetection에 적용
- 우리 task의 Input 이미지는 정사각형으로 들어오기 때문에 auto augmentation을 정사각형 모양으로 설정해 줌 -> 성능향상!
- public mAP가 0.4999, 0.4903, 0.4848인 세 개의 모델을 wbf 앙상블을 통해 0.5427까지 올림
---
## 아쉬운 점
- 대회가 2주로 너무 짧았던 점
- bifpn을 커스텀하여 적용해보지 못한 점
---
## 팀원들과의 실험기록
https://www.notion.so/7f392074137f4ec2bdd4d1b8618c34ac
