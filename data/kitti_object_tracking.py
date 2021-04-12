""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017

Modified by Yurong You
Date: June 2019

Modified by Yuwei Yan
Date: March 2021
"""
from __future__ import print_function

import os
import data.kitti_util_tracking as utils
import numpy as np
import PIL.Image as Image

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def get_image_ptc(ptc, calib, img):
    img_height, img_width, _ = img.shape
    _, _, img_fov_inds = get_lidar_in_image_fov(
        ptc[:, :3], calib, 0, 0, img_width - 1, img_height - 1, True)
    ptc = ptc[img_fov_inds]

    return ptc


def gen_depth_map(ptc, calib, img):
    ptc = get_image_ptc(ptc, calib, img)
    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) - 1
    ptc_image = calib.project_velo_to_image3(ptc[:, :3])

    ptc_2d = np.round(ptc_image[:, :2]).astype(np.int32)
    depth_info = calib.project_velo_to_rect(ptc[:, :3])
    depth_map[ptc_2d[:, 1], ptc_2d[:, 0]] = depth_info[:, 2]
    return depth_map


class kitti_object_tracking(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir,
                 split='training', lidar_dir='velodyne',
                 label_dir='label_02', calib_dir='calib',
                 image2_dir='image_02', image3_dir='image_03'):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.sequence = 21
        elif split == 'testing':
            self.sequence = 29
        else:
            print('Unknown split: %s' % split)
            exit(-1)

        self.image2_dir = os.path.join(self.split_dir, image2_dir)
        self.image3_dir = os.path.join(self.split_dir, image3_dir)
        self.label_dir = os.path.join(self.split_dir, label_dir)
        self.calib_dir = os.path.join(self.split_dir, calib_dir)
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)

    def __len__(self):
        return self.sequence

    def get_image_path(self, sequence, idx, is_left=True):
        if is_left:
            return os.path.join(self.image2_dir, sequence, '%06d.png' % (idx))
        else:
            return os.path.join(self.image3_dir, sequence, '%06d.png' % (idx))

    def get_image(self, sequence, idx):
        img_filename = os.path.join(self.image2_dir, '%04d' % sequence, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_image2(self, sequence, idx):
        img_filename = os.path.join(self.image2_dir, '%04d' % sequence, '%06d.png' % idx)
        return Image.open(img_filename).convert('RGB')

    def get_image3(self, sequence, idx):
        img_filename = os.path.join(self.image3_dir, '%04d' % sequence, '%06d.png' % (idx))
        return Image.open(img_filename).convert('RGB')

    def get_lidar(self, sequence, idx):
        lidar_filename = os.path.join(self.lidar_dir, '%04d' % sequence, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, sequence):
        calib_filename = os.path.join(self.calib_dir, '{:04d}.txt'.format(sequence))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, sequence):
        assert (self.split == 'training')
        label_filename = os.path.join(self.label_dir, '{:04d}.txt'.format(sequence))
        return utils.read_label(label_filename)

    def get_depth_map(self, sequence, idx):
        # print(self.get_image(idx).shape)
        # print(self.get_image1(idx).size)
        return gen_depth_map(self.get_lidar(sequence, idx), self.get_calibration(sequence),
                             self.get_image(sequence, idx))

    def get_top_down(self, idx):
        pass


# fov表示相机的视场角
def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_rect_in_image_fov(pc_rect, calib, xmin, ymin, xmax, ymax,
                          return_more=False, clip_distance=2.0):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_rect_to_image(pc_rect)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_rect[:, 2] > clip_distance)
    imgfov_pc = pc_rect[fov_inds, :]
    if return_more:
        return imgfov_pc, pts_2d, fov_inds
    else:
        return imgfov_pc
