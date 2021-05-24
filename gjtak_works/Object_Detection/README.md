# P-stage3 Object Detection 한사랑개발회 탁금지 실험일지

## 실험 일지

<details>
    <summary>2021-05 3주차(Object Detection 1주차)</summary>
        <h4>05-10-Mon</h4>
        <p> - mmdetection 기본 baseline code 실행(faster rcnn-resnet50)</p>
        <h4>05-11-Tue</h4>
        <p> - git branch 생성(gjtak_branch) & code directory 포함시킴</p>
        <h4>05-12-Wed</h4>
        <p> - jupyter notebook 수정(wandb 추가)</p>
        <p> - wandb(pstage3_det) 작동 test - faster_rcnn_r50_fpn_1x_coco.py 이용</p>
        <p> - faster_rcnn_hrnetv2p_w40_2x_coco.py basic code 실행</p>
        <h4>05-13-Thur</h4>
        <p> - fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco.py 실행. 결과가 이전보다 좋지 않음. hyper parameter 조정이 필요해보임</p>
        <p> - Cumstomized coding 계획 수립(Dataset ~ Train ~ Test 까지)</p>
        <p> - cascade_rcnn_hrnetv2p_w40_20e_coco.py 실행.<del> 오류 발생. 일단 보류</del> 해결. 용량 문제 였음.</p>
        <p> - Yolov4 code 작성 시작. Yolov3 code 참조.</p>
        <p> - Yolov4 code 작성 시작</p>
        <h4>05-14-Fri</h4>
        <p> - <del>DetectoRS(ResNeXt-101-32x4d, single scale/multi scale) 코딩 시작</del>mmdetection 보류. 원본 github보면서 구조 분석 진행</p>
        <p> - train한 model로 inference한 결과를 볼 수 있게하는 matplotlib coding 시작. 현재 mmdetection에서 제공하는 코드는 customed class를 인식하지 못함</p>
</details>


<details>
    <summary>2021-05 4주차(Object Detection 2주차)</summary>
</details>


## 실험 결과 정리

| Date       | Model Name   | Backbone | Config file link                                             | WanDB Link                                                   | Last bbox mAP50(val) | LB score(mAP50) | ETC                                                          |
| ---------- | ------------ | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- | --------------- | ------------------------------------------------------------ |
| 2021-05-12 | Faster RCNN  | resnet50 | [faster_rcnn_r50_fpn_1x_coco config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/faster_rcnn_r50_fpn_1x_coco/config.json) | [faster_rcnn_r50_fpn_1x_coco](https://wandb.ai/pstage3_det/gjtak/runs/11ckhm1c?workspace=user-atica) | 0.313                | 0.3663          | basic tutorial code                                          |
| 2021-05-12 | Faster RCNN  | hrnet    | [faster_rcnn_hrnetv2p_w40_2x_coco config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/faster_rcnn_hrnetv2p_w40_2x_coco_24/config.json) | [faster_rcnn_hrnetv2p_w40_2x_coco](https://wandb.ai/pstage3_det/gjtak/runs/2gm7klxk?workspace=user-atica) | 0.341                | 0.3975          | basic tutorial code                                          |
| 2021-05-13 | FCOS         | hrnet    | [fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_24/config.json) | [fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco](https://wandb.ai/pstage3_det/gjtak/runs/i5vlne35?workspace=user-atica) | 0.19                 | 0.2244          | basic tutorial code                                          |
| 2021-05-13 | Cascade RCNN | hrnet    | [cascade_rcnn_hrnetv2p_w40_20e_coco config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/cascade_rcnn_hrnetv2p_w40_20e_coco_24/config.json) | [cascade_rcnn_hrnetv2p_w40_20e_coco](https://wandb.ai/pstage3_det/gjtak/runs/gk8nr112?workspace=user-atica) | 0.345                | 0.3858          | basic tutorial code<br>faster_rcnn_hrnetv2p_w40_2x_coco 보다 살짝 높은 val 점수를 기록했는데 LB는 하락함 |
| 2021-05-18 | Faster RCNN  | hrnet    | [faster_rcnn_hrnetv2p_w40_2x_coco(autoAug)_30 config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/faster_rcnn_hrnetv2p_w40_2x_coco_(autoAug)_30/config.json) | [faster_rcnn_hrnetv2p_w40_2x_coco(autoAug)_30](https://wandb.ai/pstage3_det/gjtak/runs/21twowvl?workspace=user-atica) | 0.337                | 0.3987          | auto augmentation 적용                                       |
| 2021-05-20 | Faster RCNN  | hrnet    | [faster_rcnn_hrnetv2p_w40_2x_coco_psudoLabeling_24 config(json)](https://github.com/bcaitech1/p3-ims-obd-hansarang/blob/main/gjtak_works/Object%20Detection/code/mmdetection_trash/work_dirs/faster_rcnn_hrnetv2p_w40_2x_coco_psudoLabeling_24/config.json) | [faster_rcnn_hrnetv2p_w40_2x_coco_psudoLabeling_24](https://wandb.ai/pstage3_det/gjtak/runs/1cfgufn7?workspace=user-atica) | 0.335                | 0.1676          | 다른 조원의 pseudo labeling json 파일을 이용해서 train 실행  |

