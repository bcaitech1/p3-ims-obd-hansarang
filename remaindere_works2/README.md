P3 stage  
Image Detection  
1일차 : Baseline 코드 및 Swin-Transformer / ViT 논문 연구  
2일차 : Baseline 코드와 Swin-Transformer 코드 참고하여 현재 task에서 사용할 수 있도록 수정중 + 모델과 백본에 대한 이해가 필요하여 공부중  
3일차 : Baseline 코드에 Swin-Transformer 이식 및 WanDB 사용 방법 숙지  
4일차 : Cascase Mask RCNN + Swin + FPN 실험 진행중    
5일차 : Colab에서도 실험할 수 있도록 실험환경 조성 중, AugoAug 활성화 실험 중, Imagenet11K로 finetune 된 ckpt에서부터 실험 중, lr 및 scheduler, Weight Decay 재설정    
6일차 : Base size 모델 최적 파라미터 / LR 및 scheduler 조정 / Autoaug 최적 resize 탐색 / Anchor Box 추가 실험 / wbf 탐구   
7일차 : 휴식    
8일차 : Backbone 에서 Model로의 Proj Dropout Rate 조절을 통하여 성능 향상 꾀함  
9일차 : HTC model을 적용   
9일차 : HTC model을 튜닝, instaboost 적용, TTA flip 적용, Kfold 학습 시작   
=========References========     
Model  
----two stage----  
rcnn : https://arxiv.org/pdf/1311.2524.pdf  
fast rcnn : https://arxiv.org/pdf/1504.08083.pdf  
faster rcnn : https://arxiv.org/pdf/1506.01497.pdf  
mask rcnn : https://arxiv.org/pdf/1703.06870.pdf  
----multi stage----   
cascade mask rcnn : https://arxiv.org/pdf/1712.00726v1.pdf  
hybrid task cascade : https://arxiv.org/pdf/1901.07518v2.pdf  
  
Backbone  
AN IMAGE IS WORTH 16X16 WORDS: https://arxiv.org/pdf/2010.11929.pdf    
DeiT : https://arxiv.org/pdf/2012.12877.pdf     
Swin Transformer : https://arxiv.org/pdf/2103.14030.pdf     
  
Github  
Swin-Transformer official : github https://github.com/microsoft/Swin-Transformer      
Swin-Transformer on mmdetection implements : https://github.com/SwinTransformer/Swin-Transformer-Object-Detection     
  
