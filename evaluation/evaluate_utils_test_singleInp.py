# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import os.path
from PIL import Image
import json
import torch
import scipy.io as sio
# import data.custom_transforms as tr

from utils.utils import get_output, get_output_val, mkdir_if_missing
from torchvision import transforms
# from data import custom_transforms as tr


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p):
        self.database = p['train_db_name']
        self.tasks = p.TASKS.NAMES
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert(set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)    
    mtl_performance = 0.0
    semseg_performance = 0.0
    
    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]
        
        if task == 'depth': # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']: # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']
            semseg_performance += (mtl['mIoU'] - stl['mIoU'])

        elif task == 'normals': # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']

        elif task == 'edge': # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks, semseg_performance


def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter()

    elif task == 'edge': # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'])

    else:
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = p.TASKS.NAMES
    
    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False
        
        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False
            # if current['depth']['silog'] < reference['depth']['silog']:
            #     print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['silog'], current['depth']['silog']))
            #     improvement = True
            # else:
            #     print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['silog'], current['depth']['silog']))
            #     improvement = False
            
        
        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True
            
            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False


    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = False

       # if current['semseg_performance'] > reference['semseg_performance']:
       #     print('New best SEMSEG-MTI model %.2f -> %.2f' %(100*reference['semseg_performance'], 100*current['semseg_performance']))
       #     improvement = True

       # else:
       #     print('No new best SEMSEG-MTI model %.2f -> %.2f' %(100*reference['semseg_performance'], 100*current['semseg_performance']))
       #     improvement = False     
       
       
       
        # if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
        #     print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
        #     improvement = True
        # else:
        #     print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
        #     improvement = False
       
       
        # if current['depth']['silog'] < reference['depth']['silog']:
        #     print('New best depth model %.2f -> %.2f' %(reference['depth']['silog'], current['depth']['silog']))
        #     improvement = True
        # else:
        #     print('No new best depth model %.2f -> %.2f' %(reference['depth']['silog'], current['depth']['silog']))
        #     improvement = False
       
       
            
            

    if improvement: # Return result
        return True, current

    else:
        return False, reference


@torch.no_grad()
def eval_model(p, val_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        lidars = batch['lidar'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
        # output = model(images)
        # output = model(lidars)
        output = model([images, lidars])
        
        # Measure performance
        performance_meter.update({t: get_output(output[t], t) for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose = True)
    return eval_results


@torch.no_grad()
def save_model_predictions(p, inputPath, img_size, model):
    """ Save model predictions for all tasks """

    print('Save model predictions to {}'.format(p['save_dir']))
    model.eval()
    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    
    transforms_val = ([transforms.ToTensor(),
                       # tr.Normalize([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])])
                         # tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                         transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])])
    transforms_val = transforms.Compose(transforms_val)
    
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    with torch.no_grad():
        
        sample = 'uu_000050'
        path = os.path.join(os.getcwd())
        inputs = np.array(Image.open(os.path.join(path + inputPath + 'image_2_split/' + sample + '.png')).convert('RGB')).astype(np.float32)

        lidars = np.array(Image.open(os.path.join(path + inputPath + 'ADI/rgb_trainSplit/transDepth/tr2/' + sample + '.png')).convert('RGB')).astype(np.float32)
        # lidars = cv2.imread(os.path.join(path + inputPath + 'ADI/rgb/transDepth/tr2/' + sample), cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs_res = cv2.resize(inputs, img_size)
        
        lidars = lidars / 128.
        lidars = lidars - np.mean(lidars[lidars>0]) 
        lidars_res = cv2.resize(lidars, img_size)

        inputs_res = transforms_val(inputs_res)
        lidars_res = torch.from_numpy(lidars_res)

       
        inputs_ = inputs_res[None,:,:,:]
        
        lidars_ = lidars_res.transpose(2, 0)
        lidars_ = lidars_.transpose(2, 1)
        lidars_ = lidars_[None,:,:,:]
        
        # output = model(inputs)
        # output = model(lidars)
        output = model([inputs_, lidars_])
 
        for task in p.TASKS.NAMES:
            output_task = get_output_val(output[task], task).cpu().data.numpy()
            fname = sample
            im_name_splits = fname.split('_')
            # cv2.imwrite('./outputs/results/' + im_name_splits[0] + '_road_' + im_name_splits[1] + '.png', output_fg)
            output_task_s = output_task*255
            output_task_s[output_task_s > 255] = 255
            result_d = cv2.resize(output_task[0,:,:], dsize=(int(inputs.shape[1]), int(inputs.shape[0])), interpolation=p.TASKS.INFER_FLAGVALS[task])
            result_s = cv2.resize(output_task_s[0,:,:], dsize=(int(inputs.shape[1]), int(inputs.shape[0])), interpolation=p.TASKS.INFER_FLAGVALS[task])

            if task == 'depth':
                sio.savemat(os.path.join(save_dirs[task], im_name_splits[0] + '_road_' + im_name_splits[1] + '.mat'), {'depth': result_d})
                # imageio.imwrite(os.path.join(save_dirs[task],im_name_splits[0] + '_road_' + im_name_splits[1]), (1*result_d).astype(np.float32))
    
            else:
                imageio.imwrite(os.path.join(save_dirs[task], im_name_splits[0] + '_road_' + im_name_splits[1] + '.png'), result_s.astype(np.uint8))
                    
                    
def eval_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir """
    save_dir = p['save_dir'] 
    results = {}
 
    if 'edge' in p.TASKS.NAMES: 
        from evaluation.eval_edge import eval_edge_predictions
        results['edge'] = eval_edge_predictions(p, database=p['val_db_name'],
                             save_dir=save_dir)
    
    if 'semseg' in p.TASKS.NAMES:
        from evaluation.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['val_db_name'],
                              save_dir=save_dir, overfit=p.overfit)
    
    if 'human_parts' in p.TASKS.NAMES: 
        from evaluation.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['val_db_name'],
                                   save_dir=save_dir, overfit=p.overfit)

    if 'normals' in p.TASKS.NAMES:
        from evaluation.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['val_db_name'],
                               save_dir=save_dir, overfit=p.overfit)

    if 'sal' in p.TASKS.NAMES:
        from evaluation.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['val_db_name'],
                           save_dir=save_dir, overfit=p.overfit)

    if 'depth' in p.TASKS.NAMES:
        from evaluation.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['val_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                 single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'], results['semseg_performance'] = calculate_multi_task_performance(results, single_task_test_dict)            
        print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results
