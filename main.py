import torch
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
from network import dla34
from loss.run import lossBuilder
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.meter import AverageMeter
from utils.model import *
import time


def main():
    # 数据集信息
    train_root = 'G:\\KITTI\\tracking'
    kitti_object = kitti_object_tracking(train_root, split='training')
    log_dir = './output/out'
    model_log_dir = './output/model'
    epoch_size = 5
    # epoch boundary used for log model
    boundary = 2 * epoch_size

    # 模型构建
    heads = {'hm': 3, 'dim': 3, 'dep': 1, 'rot': 8}
    model = dla34.DLASeg('dla34', heads, 'model/')
    lossModel = lossBuilder()
    optimizer = Adam(model.parameters(), 1e-4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    # Fold num
    K = 7

    # loss accumulation
    accumulation_step = 4

    # writer
    writer = SummaryWriter(log_dir=log_dir, comment='3D_Tracker{}'.format(int(time.time())))

    # GradScaler
    # scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)

    # start training/val
    best_val_loss = float('inf')
    for i in range(K):
        train_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i+1, K=K, typ='train')
        val_dataset = tracking_dataset(kitti_object, root_dir=train_root, ki=i+1, K=K, typ='val')
        train_loader = DataLoader(train_dataset, batch_size=3, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=3, drop_last=True)
        for epoch in range(epoch_size):
            loss_stat = {'train': AverageMeter(), 'val': AverageMeter()}
            loss_stat_heads = {head: AverageMeter() for head in heads}

            model.train()
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                print("train ----> fold:{}, epoch:{}, batch:{}------------------------------"
                      .format(i, epoch, idx))
                for key in batch:
                    batch[key] = batch[key].to(device)
                image2 = batch['image2']
                image3 = batch['image3']
                stereo_img = torch.cat((image2, image3), dim=1).float()
                # if idx == 0:
                #     idx1 = stereo_img
                #     bat1 = batch
                # output
                # model_seq = nn.Sequential(model, lossModel)
                with autocast(enabled=True):
                    output = model(stereo_img)
                    # output = checkpoint(model.forward, stereo_img)
                    # output = checkpoint_sequential(model_seq, segments=2, input=stereo_img)
                    # loss
                    total_loss, losses = lossModel(output, batch)
                    loss_mean = total_loss.mean()
                    loss_stat['train'].update(loss_mean)
                    for head in heads:
                        loss_stat_heads[head].update(losses[head].mean())
                    # backward in autocast is not recommended, but as we use
                    # checkpoint, if don't there are TypeError
                    loss_mean /= accumulation_step
                    # scaler.scale(loss_mean).backward()  # scaler the loss and backward
                    loss_mean.backward()
                if (idx + 1) % accumulation_step == 0:
                    # ensure weather the optimizer make sence
                    # with torch.no_grad():
                    #     output = model(idx1)
                    #     total_loss, losses = lossModel(output, bat1)
                    #     loss1 = total_loss.mean()
                    # print("loss for images idx1:{}".format(loss1))
                    # scaler.step(optimizer)  # wrapper of optimizer
                    # scaler.update()  # update the parameter of the scaler
                    # print("scale:{}".format(scaler.get_scale()))
                    optimizer.step()
            writer.add_scalar('train total loss', loss_stat['train'].avg, i * epoch_size + epoch)
            for head in heads:
                writer.add_scalar('{} train loss'.format(head), loss_stat_heads[head].avg, i * epoch_size + epoch)

            model.eval()
            for idx, batch in enumerate(val_loader):
                print("val ----> fold:{}, epoch:{}, batch:{}------------------------------"
                      .format(i, epoch, idx))
                for key in batch:
                    batch[key] = batch[key].to(device)
                image2 = batch['image2']
                image3 = batch['image3']
                stereo_img = torch.cat((image2, image3), dim=1).float()
                with torch.no_grad():
                    output = model(stereo_img)
                    total_loss, losses = lossModel(output, batch)
                    loss_mean = total_loss.mean()
                    if epoch_size * i + epoch > boundary:
                        if loss_mean < best_val_loss:
                            best_val_loss = loss_mean
                            save_model(model_log_dir, epoch_size * i + epoch, model=model, optimizer=optimizer)
                    loss_stat['val'].update(loss_mean)
                    for head in heads:
                        loss_stat_heads[head].update(losses[head].mean())
                    loss_mean /= accumulation_step
            writer.add_scalar('val total loss', loss_stat['val'].avg, i * epoch_size + epoch)
            for head in heads:
                writer.add_scalar('{} val loss'.format(head), loss_stat_heads[head].avg, i * epoch_size + epoch)
        save_model(path=model_log_dir, epoch=epoch_size*(i+1), model=model, optimizer=optimizer)
    writer.export_scalars_to_json('./all_scalars.json')
    writer.close()


if __name__ == '__main__':
    main()
