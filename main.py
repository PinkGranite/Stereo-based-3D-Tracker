import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.nn.functional as F
from data.kitti_util_tracking import *
from data.kitti_object_tracking import *
from data.tracking_dataset import tracking_dataset
from torchvision import transforms
from network import dla34
from loss.run import lossBuilder


def main():
    # 数据集信息
    train_root = 'G:\\KITTI\\tracking'
    kitti_object = kitti_object_tracking(train_root, split='training')

    # 模型构建
    heads = {'hm': 3, 'dim': 3, 'dep': 1, 'orientation': 8}
    model = dla34.DLASeg('dla34', heads, 'model/')
    lossModel = lossBuilder()
    optimizer = Adam(model.parameters(), 1e-4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    K = 7
    # loss accumulation
    accumulation_step = 4
    for i in range(K):
        train_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i, K=K, typ='train')
        val_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i, K=K, typ='val')
        train_loader = DataLoader(train_dataset, batch_size=3, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, drop_last=True)
        for epoch in range(50):
            # 50个epoch
            for idx, batch in enumerate(train_loader):
                print("batch{}.....".format(idx))
                for key in batch:
                    batch[key] = batch[key].to(device)
                image2 = batch['image2']
                image3 = batch['image3']
                stereo_img = torch.cat((image2, image3), dim=1).float()
                # stereo_img = Variable(stereo_img, requires_grad=True)
                # output
                model_seq = nn.Sequential(model, lossModel)
                with autocast(enabled=True):
                    output = model(stereo_img)
                    # output = checkpoint(model.forward, stereo_img)
                    # output = checkpoint_sequential(model_seq, segments=2, input=stereo_img)
                    # loss
                    total_loss, losses = lossModel(output, batch)
                    loss_mean = total_loss.mean()
                    # backward in autocast is not recommended, but as we use
                    # checkpoint, if don't there are TypeError
                    loss_mean /= accumulation_step
                    loss_mean.backward()
                if (idx+1) % accumulation_step == 0:
                    optimizer.zero_grad()
                    optimizer.step()
                    print('success !!')
                del output, total_loss, losses, loss_mean
                
            break
        break


if __name__ == '__main__':
    main()
