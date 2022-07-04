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

from ptsemseg.utils import recursive_glob
from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
from data.google_drive import download_file_from_google_drive

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
        
        self.mean = np.array(self.mean_rgb)
        
        self.retname = retname
        
        if split == 'train': 
            # Original Images & Lidar depth Images
            self.im_ids = []
            self.images = []
            _image_dir = os.path.join( self.root, 'image_2_split')
            self.lidar_ids = []
            self.lidars = []
            # _lidar_dir = os.path.join( self.root, 'ADI/gray_trainSplit')
            _lidar_dir = os.path.join( self.root, 'ADI/rgb_trainSplit/transDepth/tr1')
            
            # Edge Detection
            self.do_edge = do_edge
            self.edges = []
            _edge_gt_dir = os.path.join( self.root, 'edge')
    
            # Semantic segmentation
            self.do_semseg = do_semseg
            self.semsegs = []
            _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split')
    
            # Surface Normals
            self.do_normals = do_normals
            self.normals = []
            _normal_gt_dir = os.path.join( self.root, 'normal')
    
            # Depth
            self.do_depth = do_depth
            self.depths = []
            _depth_gt_dir = os.path.join( self.root, 'proj_depth/groundtruth/image_2_split')
            
        else:
            # Original Images & Lidar depth Images
            self.im_ids = []
            self.images = []
            _image_dir = os.path.join( self.root, 'image_2_split')
            self.lidar_ids = []
            self.lidars = []
            _lidar_dir = os.path.join( self.root, 'ADI/rgb_testSplit/transDepth/tr1')
            
            # Edge Detection
            self.do_edge = do_edge
            self.edges = []
            _edge_gt_dir = os.path.join( self.root, 'edge')
    
            # Semantic segmentation
            self.do_semseg = do_semseg
            self.semsegs = []
            _semseg_gt_dir = os.path.join( self.root, 'gt_image_2_split')
    
            # Surface Normals
            self.do_normals = do_normals
            self.normals = []
            _normal_gt_dir = os.path.join( self.root, 'normal')
    
            # Depth
            self.do_depth = do_depth
            self.depths = []
            _depth_gt_dir = os.path.join( self.root, 'proj_depth/groundtruth/image_2_split')
            
            
       

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

        # if self.do_edge:
        #     assert (len(self.images) == len(self.edges))
        # if self.do_semseg:
        #     assert (len(self.images) == len(self.semsegs))
        # if self.do_depth:
        #     assert (len(self.images) == len(self.depths))
        # if self.do_normals:
        #     assert (len(self.images) == len(self.normals))

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
        # _img = _img.astype(np.float64)
        _img -= self.mean
        img = cv2.resize(_img, self.img_size)
        img = img.transpose(2, 0, 1)
        sample['image'] = img
        
        _lidar = self._load_lidar(index)
        _lidar = _lidar / 128.
        _lidar = _lidar - np.mean(_lidar[_lidar>0]) 
        _lidar = cv2.resize(_lidar, self.img_size)
        _lidar = _lidar.transpose(2, 0, 1)
        sample['lidar'] = _lidar

        if self.do_edge:
            _edge = self._load_edge(index)
            # if _edge.shape != _img.shape[:2]:
            #     _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            lbl_tmp = self._load_semseg(index)
                 
            # lbl_tmp = np.array(lbl_tmp, dtype=np.uint8)
            # _semseg = 255 + np.zeros( (_img.shape[0], _img.shape[1]), np.uint8)
            # _semseg[lbl_tmp[:,:,2] > 0] = 1
            # _semseg[(lbl_tmp[:,:,2] > 0) & (lbl_tmp[:,:,0] == 0)] = 0
            
            
         
            label_array = np.asarray(lbl_tmp)
            # Current class label encoding
            non_road_label = np.array([255,0,0])
            # road_label = np.array([255,0,255])
            # other_road_label = np.array([0,0,0])
            
            # Create binary class label (1=road, 0=not road) by inverting non-road label
            # _semseg = (1-np.all(label_array==non_road_label, axis=2) - np.all(label_array==other_road_label, axis=2)).astype(np.float32)
            _semseg = (1-np.all(label_array==non_road_label, axis=2)).astype(np.float32)
            # _semseg =  label_array[:,:,2]         
            # Binary label image
            # _semseg = pil.Image.fromarray(_semseg*255)
            # _semseg.show()
        
            
            _semseg = cv2.resize(_semseg, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            # _semseg = _semseg.astype(np.float32)
            # if _semseg.shape != _img.shape[:2]:
            #     print('RESHAPE SEMSEG')
            #     _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST) 
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            
            _normals = _normals.astype(np.float64) / 65535.
            _normals = cv2.resize(_normals, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            # if _normals.shape[:2] != _img.shape[:2]:
            #     _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            
            _depth = _depth.astype(np.float64) / 255.
            # _depth = _depth - np.mean(_depth[_depth>0]) 
            
            _depth = cv2.resize(_depth, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            # if _depth.shape[:2] != _img.shape[:2]:
            #     print('RESHAPE DEPTH')
            #     _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(im_idss),
                              'im_size': (img.shape[1], img.shape[2])}

        # if self.transform is not None:
        #     sample = self.transform(sample)

        return sample
        

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_lidar(self, index):
        _lidar = np.array(Image.open(self.lidars[index]).convert('RGB')).astype(np.float32)
        # _lidar = np.array(Image.open(self.lidars[index])).astype(np.uint8)
        # _lidar = cv2.imread(os.path.join(self.lidars[index] + '.png'), cv2.IMREAD_UNCHANGED)
        # _lidar = np.array(_lidar, dtype=np.uint8)
        return _lidar
    
    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        # _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        # _semseg = cv2.imread(self.semsegs[index], cv2.IMREAD_UNCHANGED)
        _semseg = pil.Image.open(self.semsegs[index])
        # _semseg.show()
        # _semseg[_semseg == 0] = 256
        # _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.array(Image.open(self.depths[index])).astype(np.float16)
        # _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.array(Image.open(self.normals[index])).astype(np.float16)
        # _normals = np.load(self.normals[index])
        return _normals

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(MyPath.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

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
