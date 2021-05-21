import argparse
import os
from importlib import import_module
import numpy as np
from tqdm import tqdm

import collections
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import SegNet, DeepLabV3Plus
from dataset import CustomDataLoader, BaseAugmentation


def load_model(saved_model, num_classes, device, epoch):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls()

    if epoch == 0:
        model_path = os.path.join(saved_model, '0_epoch11_0.3385_0.9851_best.pth')
    elif epoch == 1:
        model_path = os.path.join(saved_model, '1_epoch3_0.3426_0.9601_best.pth')
    elif epoch == 2:
        model_path = os.path.join(saved_model, '2_epoch14_0.3447_0.9194_best.pth')
    elif epoch == 3:
        model_path = os.path.join(saved_model, '3_epoch12_0.3454_0.9217_best.pth')
    elif epoch == 4:
        model_path = os.path.join(saved_model, '4_epoch13_0.3456_0.93_best.pth') 

    if epoch == -1:
        model_path = os.path.join(saved_model, '0_epoch11_0.3385_0.9851_best.pth')     
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    model_dir = model_dir + '/' + 'n_0_epoch1_0.9433.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    aux_params=dict(
            pooling='avg',
            dropout=args.dropout,
            activation=None,
            classes=12,
        )
    model = DeepLabV3Plus(args.backbone, aux_params)

    if args.multi == 0:
        
         # best model 불러오기
        checkpoint = torch.load(model_dir, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)   


        # -- augmentation
        test_transform = A.Compose([
                           A.transforms.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
                           ToTensorV2(),
                           #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                           ])

        test_dataset = CustomDataLoader(data_dir=data_dir, mode='test', transform=test_transform)
        
        # -- data_loader
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)

        size = 256
        transform = A.Compose([A.Resize(256, 256)])
        print('Start prediction.')
        model.eval()
        
        file_name_list = []
        preds_array = np.empty((0, size*size), dtype=np.long)
        
        with torch.no_grad():
            for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

                # inference (512 x 512)
                outs = model(torch.stack(imgs).to(device))
                (a, b) = outs
                oms = torch.argmax(a.squeeze(), dim=1).detach().cpu().numpy()
                
                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)

                oms = np.array(temp_mask)
                
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                preds_array = np.vstack((preds_array, oms))
                
                file_name_list.append([i['file_name'] for i in image_infos])
        print("End prediction.")
        file_names = [y for x in file_name_list for y in x]
    
    elif args.multi == 1:
        print('not ready yet')
    
    np.save('/opt/ml/code/submission/pred_sce2', preds_array)
    
    
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)
    # submission.csv로 저장
    submission.to_csv(f"./submission/Baseline_{args.name}.csv", index=False)
    print("submission ready")
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 16)')
    #parser.add_argument('--resize', type=tuple, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    #parser.add_argument('--model', type=str, default='MyModel2', help='model type (default: BaseModel)')
    parser.add_argument('--backbone', type=str, default='timm-efficientnet-b3')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--multi', type=int, default=0, help='single_model(default:0), 5fold ensemble:1')
    parser.add_argument('--name', default='DeepLabV3Plus-timm-efficientnet-b3', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '../input/data/test.json'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './saved'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './submission'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args) 
