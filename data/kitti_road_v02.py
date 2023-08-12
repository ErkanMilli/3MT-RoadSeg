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
import PIL as pil
from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

import torch
from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
from utils.utils import recursive_glob
from data.google_drive import download_file_from_google_drive

class KITTIRoad(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """    

    def __init__(self,
                 root=MyPath.db_root_dir('KITTIRoad'),
                 # download=True,
                 split='val',
                 transform=None,
                 img_size=(1280, 384),
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):
        
        if split == 'train':
            self.root = os.path.join(root, 'training')
        else:
            self.root = os.path.join(root, 'testing')

        # if download:
        #     self._download()

        self.transform = transform
        
        self.img_size = img_size

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        
        # self.mean = np.array(self.mean_rgb)
        # self.std = np.array(self.std_rgb)
        
        self.retname = retname
        
        if split == 'train': 
            # Original Images & Lidar depth Images
            self.im_ids = []
            self.images = []
            _image_dir = os.path.join( self.root, 'image_2_split')
            # _image_dir = os.path.join( self.root, 'image_2_split_cropped')
            # _image_dir = os.path.join( self.root, 'image_2')
            # _image_dir = os.path.join( self.root, 'image_2_cropped')
            self.lidar_ids = []
            self.lidars = []
            # _lidar_dir = os.path.join( self.root, 'ADI/gray')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_cropped')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_trainSplit')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_trainSplit_cropped')
            _lidar_dir = os.path.join( self.root, 'ADI/rgb_trainSplit/transDepth/tr2')
            # _lidar_dir = os.path.join( self.root, 'ADI/rgb_trainSplit/transDepth/tr2_cropped')
            # _lidar_dir = os.path.join( self.root, 'ADI/rgb/transDepth/tr2')
            # _lidar_dir = os.path.join( self.root, 'depth_u16_3ChnGaussian_split')
            # _lidar_dir = os.path.join( self.root, 'SNE_depthNormal_split')
            
            # Edge Detection
            self.do_edge = do_edge
            self.edges = []
            _edge_gt_dir = os.path.join( self.root, 'edge')
    
            # Semantic segmentation
            self.do_semseg = do_semseg
            self.semsegs = []
            _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split_cropped')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_cropped')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_zeroImg_trainSplit')
    
            # Surface Normals
            self.do_normals = do_normals
            self.normals = []
            # _normal_gt_dir = os.path.join( self.root, '3ChnNormal_split')
            _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_split')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_split_cropped')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_cropped')

    
            # Depth
            self.do_depth = do_depth
            self.depths = []
            # _depth_gt_dir = os.path.join( self.root, 'proj_depth/groundtruth/image_2_split')
            # _depth_gt_dir = os.path.join( self.root, 'denseDepth_split')
            _depth_gt_dir = os.path.join( self.root, 'depth_u16_split')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16_split_cropped')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16_cropped')

            
        else:
            # Original Images & Lidar depth Images
            self.im_ids = []
            self.images = []
            _image_dir = os.path.join( self.root, 'image_2_split')
            # _image_dir = os.path.join( self.root, 'image_2_split_cropped')
            # _image_dir = os.path.join( self.root, 'image_2')
            # _image_dir = os.path.join( self.root, 'image_2_cropped')
            self.lidar_ids = []
            self.lidars = []
            # _lidar_dir = os.path.join( self.root, 'ADI/gray')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_cropped')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_testSplit')
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_testSplit_cropped')
            _lidar_dir = os.path.join( self.root, 'ADI/rgb_testSplit/transDepth/tr2')
            # _lidar_dir = os.path.join( self.root, 'ADI/rgb_testSplit/transDepth/tr2_cropped')
            # _lidar_dir = os.path.join( self.root, 'ADI/rgbGaussian')
            # _lidar_dir = os.path.join( self.root, 'depth_u16_3ChnGaussian_split')
            # _lidar_dir = os.path.join( self.root, 'SNE_depthNormal_split')
            
            # Edge Detection
            self.do_edge = do_edge
            self.edges = []
            _edge_gt_dir = os.path.join( self.root, 'edge')
    
            # Semantic segmentation
            self.do_semseg = do_semseg
            self.semsegs = []
            _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split_cropped')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_zeroImg_testSplit')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_zeroImg')
            # _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_zeroImg_cropped')

        
            # Surface Normals
            self.do_normals = do_normals
            self.normals = []
            # _normal_gt_dir = os.path.join( self.root, '3ChnNormal_split')
            _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_split')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_split_cropped')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal')
            # _normal_gt_dir = os.path.join( self.root, 'SNE_depthNormal_cropped')

    
            # Depth
            self.do_depth = do_depth
            self.depths = []
            # _depth_gt_dir = os.path.join( self.root, 'proj_depth/groundtruth/image_2_split')
            # _depth_gt_dir = os.path.join( self.root, 'denseDepth_split')
            _depth_gt_dir = os.path.join( self.root, 'depth_u16_split')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16_split_cropped')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16')
            # _depth_gt_dir = os.path.join( self.root, 'depth_u16_cropped')

       
        print('Initializing dataloader for KITTIRoad {} set'.format(''.join(self.split)))

        # Images & Lidars
        self.image_files = recursive_glob(_image_dir, suffix='.png')
        self.images = sorted(self.image_files)
        
        self.lidar_files = recursive_glob(_lidar_dir, suffix='.png')
        self.lidars = sorted(self.lidar_files)
        
        # Edges
        self.edge_files = recursive_glob(_edge_gt_dir, suffix='.png')
        self.edges = sorted(self.edge_files)

        # Semantic Segmentation    
        self.semseg_files = recursive_glob(_semseg_gt_dir, suffix='.png')
        self.semsegs = sorted(self.semseg_files)

        # Surface Normals
        self.normal_files = recursive_glob(_normal_gt_dir, suffix='.png')
        self.normals = sorted(self.normal_files)
        
        # Depth Prediction
        self.depth_files = recursive_glob(_depth_gt_dir, suffix='.png')
        self.depths = sorted(self.depth_files)

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}
        
        img_path = self.images[index].rstrip()
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')
        im_idss = im_name_splits[0] + '_' + im_name_splits[1]

        _img = self._load_img(index)
        img = cv2.resize(_img, self.img_size)
       
        sample['image'] = img
        
        _lidar = self._load_lidar(index)
        _lidar = _lidar / 128.
        _lidar = _lidar - np.mean(_lidar[_lidar>0]) 
        _lidar = cv2.resize(_lidar, self.img_size)
        sample['lidar'] = _lidar
        
        if self.do_edge:
            _edge = self._load_edge(index)
            sample['edge'] = _edge

        if self.do_semseg:
            lbl_tmp = self._load_semseg(index)
            _semseg = np.zeros((int(_img.shape[0]), int(_img.shape[1])), dtype=np.uint8)
            _semseg[lbl_tmp[:, :, 2] > 0] = 1
            _semseg = cv2.resize(_semseg, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            _normals = cv2.resize(_normals, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            _depth = _depth.astype(np.float64) / 65535.
            _depth = cv2.resize(_depth, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(im_idss),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
        

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_lidar(self, index):
        _lidar = np.array(Image.open(self.lidars[index]).convert('RGB')).astype(np.float32)
        return _lidar
    
    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        _semseg = cv2.cvtColor(cv2.imread(self.semsegs[index]), cv2.COLOR_BGR2RGB)
        return _semseg

    def _load_depth(self, index):
        _depth = np.array(Image.open(self.depths[index])).astype(np.float32)
        # _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.array(Image.open(self.normals[index])).astype(np.float32)
        # _normals = np.load(self.normals[index])
        return _normals

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


def test_mt():
    import torch
    import data.custom_transforms as tr
    import  matplotlib.pyplot as plt 
    from torchvision import transforms
    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-2, 2), scales=(.75, 1.25),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_LINEAR,
                                                              'depth': cv2.INTER_LINEAR}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'normals': (512, 512),
                                                                'depth': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_LINEAR,
                                                             'depth': cv2.INTER_LINEAR}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = KITTIRoad(split='train', transform=transform, retname=True,
                      do_edge=True,
                      do_semseg=True,
                      do_normals=True,
                      do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):
        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(5)
            for k in range(len(ax_arr)):
                ax_arr[k].cla()
            ax_arr[0].imshow(np.transpose(sample['image'][j], (1,2,0)))
            ax_arr[1].imshow(sample['edge'][j,0])
            ax_arr[2].imshow(sample['semseg'][j,0]/40)
            ax_arr[3].imshow(np.transpose(sample['normals'][j], (1,2,0)))
            max_depth = torch.max(sample['depth'][j,0][sample['depth'][j,0] != 255]).item()
            ax_arr[4].imshow(sample['depth'][j,0]/max_depth) # Not ideal. Better is to show inverse depth.

            plt.show()
        break


if __name__ == '__main__':
    test_mt()
