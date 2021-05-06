import argparse
import os
from importlib import import_module
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def test(args, fold_num, checkpoint):
    # model_path = './saved/DeepLabv3_best_model(effnetb7_2).pt'
    model_path = os.path.join('saved', checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = '../input/data'
    test_path = dataset_path + '/test.json'

    test_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.25, 0.25, 0.25), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])
    test_dataset = CustomDataLoader(
        data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    size = 256
    transform = A.Compose([A.Resize(256, 256)])

    model_module = getattr(import_module("model"), args.model)
    model = model_module(n_classes=12, n_blocks=[
                         3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]).to(device)

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print('Loaded Weight From ckpt')
    print('================Inference Phase Started================')
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            # print(step)
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

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
    submission = pd.read_csv(
        './submission/sample_submission.csv', index_col=None)
    # PredictionString 대입
    np.save(f'fold{fold_num}.npy', preds_array)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                                       ignore_index=True)
    # submission.csv로 저장
    submission_path = os.path.join(
        'submission', args.submission_name+str(fold_num))
    submission.to_csv(
        submission_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--model', type=str, default='DeepLabV3',
                        help='model type (default: DeepLabV3)')
    parser.add_argument('--ckpt', type=str, default='DeepLabV3_4epoch.pt',
                        help='name of the ckpt (default: DeepLabV3_4epoch.pt)')
    parser.add_argument('--submission_name', type=str,
                        default="DeepLabV3(Effnetb7).csv")
    args = parser.parse_args()
    checkpoint_list = [
        'kfold_ensemble_fold0_25epoch_mIoU_0.6137942168203708.pt', 'kfold_ensemble_fold1_25epoch_mIoU_0.614572927543238.pt',
        'kfold_ensemble_fold2_20epoch_mIoU_0.5781823410343429.pt', 'kfold_ensemble_fold3_14epoch_mIoU_0.6281244959688194.pt', 'kfold_ensemble_fold4_21epoch_mIoU_0.5963408874797863.pt']

    for i in range(5):
        test(args, i, checkpoint_list[i])
