# 이미지 분류
> Private team mIoU: 0.6831
![image](https://user-images.githubusercontent.com/54899906/120743145-61380f80-c533-11eb-899c-2eeb09dcca81.png)
---
## 대회 개요
사진 속 픽셀에 대하여 배경인지 쓰레기의 종류인지에 따라 총 12가지 클래스로 분류하는 semantic segmentation task
- 데이터
  1. 전체 이미지 개수 : 4109장
  2. 12 class : Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  3. 이미지 크기 : (512, 512)
- 평가 방법
  - Test set의 mIoU
---
## 앙상블 전 내 모델
- model : PANnet(encoder=efficientnet-b5)
- optimizer : madgrad
- Loss : FocalLoss
- lr_scheduler : cosLR
![image](https://user-images.githubusercontent.com/54899906/120745258-bbd36a80-c537-11eb-8d67-c9cc9eff58a1.png)
---
## 나의 시도
- 데이터의 불균형이 존재하는 task라고 판단하여 focalLoss를 사용
- 팀원들과의 앙상블을 위한 코드를 작성
- 다양한 optimizer와 lr scheduler의 조합으로 PANnet의 성능 향상
---
## 아쉬운 점
- pseudo labelling이 이번 task에서 성능향상에 매우 효과적이었다는 것을 너무 늦게 알았다.
- augmentation 실험을 많이 못해봤다.
