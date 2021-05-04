import torch
from model import *
from dataset import TrashDataset
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import numpy as np
from importlib import import_module
import albumentations as A
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm
plt.rcParams['axes.grid'] = False

model_path = './saved/best.pth'
dataset_path = '../input/data'
test_path = dataset_path + '/test.json'


def collate_fn(batch):
    return tuple(zip(*batch))


use_cuda = torch.cuda.is_available()
print("PyTorch version:[%s]." % (torch.__version__))
device = torch.device('cuda' if use_cuda else 'cpu')
print("device:[%s]." % (device))
# best model 불러오기
batch_size = 9
checkpoint = torch.load(model_path, map_location=device)
model = smp.DeepLabV3Plus(
    encoder_name="timm-efficientnet-b5",
    encoder_weights="noisy-student",
    in_channels=3,
    classes=12
).to(device)
model.load_state_dict(checkpoint)
test_dataset = TrashDataset(data_dir=test_path, mode='test')
transform_module = getattr(import_module("dataset"), "NormalizedAugmentation")  # default: NormalizedAugmentation
transform = transform_module()
test_dataset.set_transform(transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)


def test(model, data_loader, device):
    size = 256
    transformer = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transformer(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array


submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

# test set에 대한 prediction
file_names, preds = test(model, test_loader, device)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append(
        {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
        ignore_index=True)

# submission.csv로 저장
submission.to_csv("./submission/resnet_ver1.csv", index=False)
