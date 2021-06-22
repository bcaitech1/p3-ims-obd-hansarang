## 공사중!
  
# 한사랑개발회 T1104_송광원  
  
## Sementic Segmentation 간이 기록  
  
04-28  
smp Library 이용하여 Model DeepLabV3+ // Backbone Eff b3 고정 이후 Loss에 대해 Test  
-> Dual Loss (Focal Loss + CE) 가 가장 성능이 우수하였음.  
  
04-29  
stepLR lr decay step 1epoch 으로 고정, weight 3가지-"imagenet", "advprop", "noisy-student" Test  
-> noisy-student 가 가장 성능이 우수하였음.  
  
04-30  
Jaccard, Tversky(tanimoto), Dice 등의 다양한 Loss function에 대한 실험 진행  
-> 기존의 Dual Loss를, 절대값을 비슷하게 맞춰준 가중치를 이용한 Loss function이 가장 우수하였음.  
  
05-01  
MADGRAD optimizer의 Weight Decay 값에 대한 실험 진행.  
-> 1e-1, 1e-2, 1e-3, 1e-4 총 4개의 값 진행, 1e-4가 가장 성능이 우수하였음.  
  
05-02  
SMP Library를 이용할 시 DeepLab 기반의 모델에서 cpu 성능 때문에 gpu 성능이 온전히 발휘되지 못하는 현상 발생함, 문제 해결을 위한 연구 진행.  
-> SMP Library를 이용하지 않고 직접 DeeplabV3를 구현하여 사용하면 해당 현상이 일어나지 않음. library 내에서 모델을 가져오는 부분에서 cpu 부하가 걸리는 듯.  
  
05-03  
DeepLabV3 모델을 직접 구현하고, backbone을 EfficientNet-b7을 이용한 상황에서 dropout에 관한 실험 진행.  
-> 실험 진행중  
  
05-04  
DeepLabV3, 즉 Decoder 단의 Conv layer 뒤와 최종 Proj layer에 dropout 을 추가  
-> Backbone에서 떼와서 Encoder의 각 부분으로 사용하는 각 layer의 뒷단에 dropout 을 추가하여 성능 향상을 꾀함.  
  
5일 저녁, 지금까지 연구한 모델의 성능이 팀 내 최고 성능을 보여주는 조원님의 모델에 비해 낮아 비슷한 성능을 보여 주는 재섭님의 코드에 현재까지의 실험으로 밝혀진 최적 loss, dropout을 적용.  
train-weight과 Weight Decay의 경우 해당 실험 진행한 날일로부터 모든 조원분들이 이미 적용하고 있었으므로 그대로 사용하였음.   
-> p = 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 값들을 Conv 뒷단 및 Encoder의 각 Layer / p = 0.1, 0.3, 0.5, 0.7 값들을 Proj 단에 붙여 실험한 결과, Encoder에 Dropout Layer 부착은 모든 값에 대해 성능이 하락하였으며 Decoder의 각 Conv Layer 뒷단에 0.05, Proj에 0.3를 사용하는 것이 성능 향상을 이루었음.  
  
05-05  
조원님이 Dice * 0.25 + Focal * 0.75가 본인 실험상 최적이었다는 말에 비교해본 결과, 이전의 Focal+CE DualLoss보다 우수한 성능을 보여주어 교체하였음.   
-> 1. 더 작은 atrous conv를 이용하여 보다 정교한 segmenation을 통한 성능 향상 꾀함.  
   2. Pesudo Labeling 을 이용하여 보다 많은 데이터셋 및 테스트 데이터에 적합하도록 모델을 학습함.  
   3. labelsmoothing * 0.75 + CE * 0.25가 기존 Dice * 0.25 + Focal * 0.75 보다 우수한 성능을 보여 교체함.  
   4. 결론 : Pesudo Labeling을 사용하였을 시에 Valid / LB 상의 성능이 둘 다 크게 성능이 향상하였음. atrous conv의 atrous 값 변경은 보다 신중하게 진행되어야 하는 실험이라 폐기됨.  
  
05-06
최종 제출을 위한 앙상블 진행.  
-> 앙상블 시 제출기준 성능지표가 상승하지 않았음. 가장 우수한 성능의 모델을 학습해내신 조원님의 최종 inference 결과를 제출하였음.  
  
실험 진행 내역 Wandb Workspace Link : https://wandb.ai/pstage12/kw_seg_p3_1  
