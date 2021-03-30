import torch
import torch.nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.kitti_util_tracking import *
from data.kitti_object_tracking import *
from data.tracking_dataset import tracking_dataset
from network import dla34


def main():
    # 数据集信息
    train_root = 'G:\\KITTI\\tracking'
    kitti_object = kitti_object_tracking(train_root, split='training')

    # 模型构建
    heads = {'hm': 3, 'dim': 3, 'dep': 1, 'orientation': 8}
    model = dla34.DLASeg('dla34', heads, 'model/')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    K = 7
    for i in range(K):
        train_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i, K=K, typ='train')
        val_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i, K=K, typ='val')
        train_loader = DataLoader(train_dataset, batch_size=2, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, drop_last=True)
        for epoch in range(50):
            # 50个epoch
            for idx, val in enumerate(train_loader):
                image2 = val['image2']
                image3 = val['image3']
                stereo_img = torch.cat((image2, image3), dim=1).float().to(device)
                with torch.no_grad():
                    output = model(stereo_img)
                print(output['hm'].shape)
                print(output['dim'].shape)
                print(output['dep'].shape)
                print(output['orientation'].shape)
                break
            break
        break


if __name__ == '__main__':
    main()
