한사랑개발회 조원 T1104 송광원 codes  

Sementic Segmentation  

04-28 : smp Library 이용하여 Model DeepLabV3+ // Backbone Eff b3 고정 이후 Loss에 대해 Test  
<        결론 : Dual Loss (Focal Loss + CE) 가 가장 성능이 우수하였음.  
04-29 : stepLR lr decay step 1epoch 으로 고정, weight 3가지-"imagenet", "advprop", "noisy-student" Test   
        결론 : noisy-student 가 가장 성능이 우수하였음.
