import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.functional as F
import cv2
import PIL.Image as Image
from data import kitti_object_tracking
from data import kitti_util_tracking
from collections import namedtuple
from utils.image import *
import math


class tracking_dataset(data.Dataset):
    # 只对三类目标进行追踪
    num_categories: int = 3
    # 输入图像的分辨率——为了适应网络结构
    default_resolution = [384, 1280]
    output_resolution = [96, 320]
    class_name = ['Pedestrian', 'Car', 'Cyclist']
    # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting', 'Tram', 'Misc', 'DontCare']
    # 这里的负值按照绝对值处理
    # 7 8 9 忽略
    cat_ids = {1: 1, 2: 2, 3: 3, 4: -2, 5: -2, 6: -1, 7: -9999, 8: -9999, 9: 0}
    # 确定一帧图像的最大主体数量
    max_objs = 50

    def __init__(self, kitti_object, root_dir, ki, K, typ='train'):
        """
            kiiti_object:kitti tracking 数据集对象
            root_dir: 数据目录
            ki: 第k折
            K: 总折数
            typ: 数据集类型，分为 train 以及 val
        """
        self.duration_frames = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145,
                                339, 1059, 837]
        self.kitti_object = kitti_object
        self.root_dir = root_dir
        self.ki = ki
        self.K = K
        self.typ = typ
        # 用于val的duration
        self.val_durations = list(range(3 * (ki - 1), 3 * ki))
        # 用于train的duration
        self.tra_durations = list(filter(lambda x: x not in self.val_durations, list(range(21))))
        # 确定数据集长度
        if typ == 'train':
            self.len = sum([self.duration_frames[x] for x in self.tra_durations])
        else:
            self.len = sum([self.duration_frames[x] for x in self.val_durations])
        self.calibrations = {}
        self.labelObjects = {}

    def __getitem__(self, index):
        sequence, index = self.get_sqAndIdx(index)
        image2, image3, calib, labels = self.get_inp(sequence, index)

        # -------------------------------------------------------------------------
        height, width = image2.size[1], image2.size[0]  # 获得图像的尺寸
        c = np.array([image2.size[1] / 2., image2.size[0] / 2.], dtype=np.float32)  # 中心点
        scale_width = self.default_resolution[1] / width  # 计算从原始图像到网络输入图像的放缩因子
        scale_height = self.default_resolution[0] / height
        # img由Image读取，已经转换为RGB格式
        #         image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        #         image3 = cv2.cvtColor(np.array(image3), cv2.COLOR_RGB2BGR)
        image2 = np.array(image2)
        image3 = np.array(image3)
        image2 = cv2.resize(image2, (self.default_resolution[1], self.default_resolution[0]),
                            interpolation=cv2.INTER_CUBIC)
        image2 = image2.transpose(2, 0, 1)
        image3 = cv2.resize(image3, (self.default_resolution[1], self.default_resolution[0]),
                            interpolation=cv2.INTER_CUBIC)
        image3 = image3.transpose(2, 0, 1)
        ret = {'image2': image2, 'image3': image3}
        self._init_ret(ret)
        num_objs = min(len(labels), self.max_objs)
        # 计算由原始图像到最终输出map的放缩因子
        scale_out = np.array((self.output_resolution[1] / width, self.output_resolution[0] / height), dtype=np.float32)
        for i in range(num_objs):
            cat_id = labels[i].type
            if cat_id > self.num_categories or cat_id < -999:
                continue
            cat_id = abs(cat_id)
            ret['cat'][i] = cat_id
            ret['mask'][i] = 1  # mask的作用是判断该位置是否是有效的
            box_centerPoint = kitti_util_tracking.project_to_image(np.array(labels[i].t).reshape(1, 3), calib.P)
            box_centerPoint = np.array([box_centerPoint[0][0] * scale_out[0], box_centerPoint[0][1] * scale_out[1]],
                                       dtype=np.int64)
            ret['ind'][i] = box_centerPoint[1] * self.output_resolution[1] + box_centerPoint[0]
            ret['dim'][i] = np.array([labels[i].h, labels[i].w, labels[i].l], dtype=np.float32)  # 将三维长宽高组织为一个array
            ret['dim'][i] = 1
            ret['dep'][i] = labels[i].t[2]
            ret['dep'][i] = 1

            # 生成heatmap
            box_2d = labels[i].box2d
            h, w = box_2d[3] - box_2d[1], box_2d[2] - box_2d[0]  # 放缩前的h, w
            h, w = h * scale_out[1], w * scale_out[0]  # 放缩后的h, w
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            # ct = np.array([(box_2d[2]-box_2d[0])/2, (box_2d[3]-box_2d[1])/2], dtype=np.float32)  # 原来的中心点
            # ct = np.array([ct[0] * scale_out[0], ct[1] * scale_out[1]], dtype=np.float32)  # 放缩后的中心点
            # ct_int = ct.astype(np.int32)
            draw_umich_gaussian(ret['hm'][cat_id - 1], box_centerPoint, radius)

            # 填充方向rot, 参见centerNet, bin based
            ret['rot_mask'][i] = 1
            ry = labels[i].ry
            if ry < np.pi / 6. or ry > 5 * np.pi / 6.:
                ret['rotbin'][i, 0] = 1
                ret['rotres'][i, 0] = ry - (-0.5 * np.pi)
            if ry > -np.pi / 6. or ry < -5 * np.pi / 6.:
                ret['rotbin'][i, 1] = 1
                ret['rotres'][i, 1] = ry - (0.5 * np.pi)

        return ret

    def __len__(self):
        return self.len

    def _init_ret(self, ret):
        # 该方法提前为输入图像的各项输入提前确定数据结构
        # 同时为GT生成空列表以待后面生成
        # hm, reg, wh, tracking, dep, rot, dim, amodel_offset
        ret['hm'] = np.zeros((self.num_categories, self.output_resolution[0], self.output_resolution[1]), np.float32)
        ret['ind'] = np.zeros(self.max_objs, dtype=np.int64)
        ret['cat'] = np.zeros(self.max_objs, dtype=np.int64)
        ret['mask'] = np.zeros(self.max_objs, dtype=np.float32)
        ret['dim'] = np.zeros((self.max_objs, 3), dtype=np.float32)
        ret['dim_mask'] = np.zeros((self.max_objs, 3), dtype=np.float32)
        ret['dep'] = np.zeros(self.max_objs, dtype=np.float32)
        ret['dep_mask'] = np.zeros(self.max_objs, dtype=np.float32)
        ret['rotbin'] = np.zeros((self.max_objs, 2), dtype=np.int64)
        ret['rotres'] = np.zeros((self.max_objs, 2), dtype=np.float32)
        ret['rot_mask'] = np.zeros(self.max_objs, dtype=np.float32)

    def get_sqAndIdx(self, index):
        """
        计算当前图像的sequence以及index
        """
        if self.typ == 'train':
            for x in self.tra_durations:
                index -= self.duration_frames[x]
                if index <= 0:
                    index += self.duration_frames[x]
                    sequence = x
                    break
        else:
            for x in self.val_durations:
                index -= self.duration_frames[x]
                if index <= 0:
                    index += self.duration_frames[x]
                    sequence = x
                    break
        return sequence, index

    def get_inp(self, sequence, index):
        # 获得左右图像
        image2 = self.kitti_object.get_image2(sequence, index)
        image3 = self.kitti_object.get_image3(sequence, index)

        # 获得calibration
        if sequence in self.calibrations.keys():
            calib = self.calibrations[sequence]
        else:
            calib = self.kitti_object.get_calibration(sequence)
            self.calibrations[sequence] = calib

        # 获得labels
        if sequence in self.labelObjects.keys():
            labels = [x for x in self.labelObjects[sequence] if x.frame_idx == index]
        else:
            self.labelObjects[sequence] = self.kitti_object.get_label_objects(sequence)
            labels = [x for x in self.labelObjects[sequence] if x.frame_idx == index]

        return image2, image3, calib, labels