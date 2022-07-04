# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2
import torch

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from torch.utils import data
from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
from data.google_drive import download_file_from_google_drive
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *
import random

class KITTIRoad(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    # GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    # FILE = 'NYUD_MT.tgz'
    
    mean_rgb = [103.939, 116.779, 123.68] # pascal mean for PSPNet and ICNet pre-trained model
    
    def __init__(self,
                 root=MyPath.db_root_dir('KITTIRoad'),
                 download=False,
                 split='val',
                 is_transform=False,
                 img_size=(320, 96),
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 phase='train'
                 ):

        self.root = root

        if download:
            self._download()

        self.is_transform = is_transform
        
        self.img_size = img_size 

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'image_2')
        
        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'proj_depth/groundtruth/image_02')
        
        self.mean = np.array(self.mean_rgb)

        if split == 'train':
            self.images_base = os.path.join(self.root, 'training', 'image_2')
            # self.images_base = os.path.join(self.root, 'training', 'image_2_split')
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI')
            self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'gray')
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'gray_trainSplit')
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'gray_refl_trainSplit')            
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'rgb', 'transDepth', 'tr1')            
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'rgb_trainSplit', 'transDepth', 'tr1')
            # self.lidar_base = os.path.join(self.root, 'training', 'ADI', 'rgb_trainSplit', 'transDepth', 'refl_tr1')
            self.annotations_base = os.path.join(self.root, 'training', 'gt_image_2')
            # self.annotations_base = os.path.join(self.root, 'training', 'gt_image_2_split')
            self.depth_gt_base = os.path.join(self.root, 'training', 'proj_depth/groundtruth/image_02')
            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.lidar_files = recursive_glob(rootdir=self.lidar_base, suffix='.png')
        else:
            self.images_base = os.path.join(self.root, 'testing', 'image_2')
            # self.images_base = os.path.join(self.root, 'testing', 'image_2_split')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI')
            self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'gray')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'gray_testSplit')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'gray_refl_testSplit')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'rgbGaussian')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'rgb_testSplit', 'transDepth', 'tr1')
            # self.lidar_base = os.path.join(self.root, 'testing', 'ADI', 'rgb_testSplit', 'transDepth', 'refl_tr1')
            # self.annotations_base = os.path.join(self.root, 'testing', 'gt_image_2')
            # self.annotations_base = os.path.join(self.root, 'testing', 'gt_image_split')        
            self.split = 'test'
        
            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.im_files = sorted(self.im_files)
            
            self.lidar_files = recursive_glob(rootdir=self.lidar_base, suffix='.png')
            self.lidar_files = sorted(self.lidar_files)
        
        self.data_size = len(self.im_files)
        self.phase = phase
        
        print("Found %d %s images" % (self.data_size, self.split))
        print("Found %d %s ADIs" % (len(self.lidar_files), self.split))
        
    def __len__(self):
        """__len__"""
        return self.data_size
    
    def im_paths(self):
        return self.im_files
    
    def __getitem__(self, index):
        """__getitem__
        
        :param index:
        """
        sample = {}
        
        img_path = self.im_files[index].rstrip()
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')
        im_idss = im_name_splits[0] + '_' + im_name_splits[1]
               
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        
        if self.retname:
            sample['meta'] = {'image': str(im_idss),
                              'im_size': (self.img_size[1], self.img_size[0])}
        
        lidar = cv2.imread(os.path.join(self.lidar_base, im_name_splits[0] + '_' + im_name_splits[1] + '.png'), cv2.IMREAD_UNCHANGED)
        lidar = np.array(lidar, dtype=np.uint8)
        
        if self.split == 'train':
            lbl_path = os.path.join(self.annotations_base,
                                    im_name_splits[0] + '_road_' + im_name_splits[1] + '.png')
        
            lbl_tmp = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            lbl_tmp = np.array(lbl_tmp, dtype=np.uint8)
                          
            lbl = 255 + np.zeros( (img.shape[0], img.shape[1]), np.uint8)
            lbl[lbl_tmp[:,:,0] > 0] = 1
            lbl[(lbl_tmp[:,:,2] > 0) & (lbl_tmp[:,:,0] == 0)] = 0
            
            
            depth_lbl_path = os.path.join(self.depth_gt_base,
                                    im_name_splits[0]  + '_' + im_name_splits[1] + '.png')
            
            depth_lbl_tmp = cv2.imread(depth_lbl_path, cv2.IMREAD_UNCHANGED)
            depth_lbl = np.array(depth_lbl_tmp, dtype=np.uint8)
            
            
            img, lidar, lbl, depth_lbl = self.transform(img, lidar, lbl, depth_lbl)
    
        
            sample['image'] = img
            sample['lidar'] = lidar
            # sample['lbl'] = lbl
            # sample['depth_lbl'] = depth_lbl
            sample['semseg'] = lbl
            sample['depth'] = depth_lbl
            # return img, lidar, lbl
            return sample
        
        else:
            tr_img = img.copy()
            tr_lidar = lidar.copy()
            tr_img, tr_lidar = self.transform(tr_img, tr_lidar)
        
            # return img, tr_img, lidar, tr_lidar
            sample['image'] = tr_img
            sample['lidar'] = tr_lidar
            # sample['lbl'] = lbl
            # sample['depth_lbl'] = depth_lbl
            # sample['semseg'] = lbl
            # sample['depth'] = depth_lbl
            # return img, lidar, lbl
            
            return sample
        
    def transform(self, img, lidar, lbl=None, depth_lbl=None):
        """transform
        
        :param img:
        :param lbl:
        """
        img = img.astype(np.float64)
        img -= self.mean
        # img -= np.mean(img, axis=(0, 1))
        
        lidar = lidar.astype(np.float64) / 128.
        lidar = lidar - np.mean(lidar[lidar>0]) 
        
        img = cv2.resize(img, self.img_size)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        lidar = cv2.resize(lidar, self.img_size)
        lidar = lidar[np.newaxis, :, :] 
        lidar = torch.from_numpy(lidar).float()
    
        # lbl = cv2.resize(lbl, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
        # lbl = torch.from_numpy(lbl).long()
        # lbl = lbl[None, :]
      
        # depth_lbl = cv2.resize(depth_lbl, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
        # depth_lbl = torch.from_numpy(depth_lbl).long()
        # depth_lbl = depth_lbl[None, :]        
        
        if lbl and depth_label is not None:
            lbl = cv2.resize(lbl, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            lbl = torch.from_numpy(lbl).long()
            lbl = lbl[None, :]
          
            depth_lbl = cv2.resize(depth_lbl, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            depth_lbl = torch.from_numpy(depth_lbl).long()
            depth_lbl = depth_lbl[None, :]
            return img, lidar, lbl, depth_lbl
        else:
            return img, lidar
           