# # 
# Authors: Erkan Milli 

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils_test import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results
from evaluation import helper                           
from evaluation.fnc_evRoad import fnc_evRoad         
# from evaluation.fnc_evRoadFast import fnc_evRoadFast                                    
from termcolor import colored
# from utils.statistic import Statistic



def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'green'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'green'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'green')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'green'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'green'))
    
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    
    
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)
    
    
    
    print(colored('Starting Final Predictions', 'green'))
    
    checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    save_model_predictions(p, val_dataloader, model)
    prob_eval_scores = fnc_evRoad(p['save_dir']+'/semseg', val_dataset)
    
    # max_F_score = 0
    # # Resume from checkpoint
    # if os.path.exists(p['checkpoint']):
    #     print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'green'))
    #     checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     model.load_state_dict(checkpoint['model'])
    #     start_epoch = checkpoint['epoch']
    #     max_F_score = checkpoint['best_result']

    # else:
    #     print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'green'))
    #     start_epoch = 0
    #     save_model_predictions(p, val_dataloader, model)
    #     prob_eval_scores = fnc_evRoad(p['save_dir']+'/semseg', val_dataset)
    #     MaxF = prob_eval_scores[1]
    #     max_F_score = MaxF
    
    # # Main loop
    # print(colored('Starting main loop', 'green'))

    # for epoch in range(start_epoch, p['epochs']):
    #     print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
    #     print(colored('-'*10, 'yellow'))

    #     # Adjust lr
    #     lr = adjust_learning_rate(p, optimizer, epoch)
    #     print('Adjusted learning rate to {:.5f}'.format(lr))

    #     # Train 
    #     print('Train ...')
    #     eval_train = train_vanilla(p, train_dataloader, model, criterion, optimizer, epoch)

    #     # Evaluate
    #         # Check if need to perform eval first
    #     if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
    #         if epoch + 1 > p['epochs'] - 10:
    #             eval_bool = True
    #         else:
    #             eval_bool = False
    #     else:
    #         eval_bool = True

    #     # Perform evaluation
    #     if eval_bool:
    #         print('Evaluate ...')
    #         save_model_predictions(p, val_dataloader, model)
    #         prob_eval_scores = fnc_evRoad(p['save_dir']+'/semseg', val_dataset)
    #         MaxF = prob_eval_scores[1]
            
    #         if MaxF > max_F_score:
    #             print('Save new best model')
    #             max_F_score = MaxF
    #             torch.save(model.state_dict(), p['best_model'])

    #             print('Checkpoint ...')
    #             torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
    #                         'epoch': epoch + 1, 'best_result': max_F_score}, p['checkpoint'])

    # # Evaluate best model at the end
    # print(colored('Evaluating best model at the end', 'green'))
    # model.load_state_dict(torch.load(p['checkpoint'])['model'])
    # save_model_predictions(p, val_dataloader, model)
    # prob_eval_scores = fnc_evRoad(p['save_dir']+'/semseg', val_dataset)
    # MaxF = prob_eval_scores[1]
    # # eval_stats = eval_all_results(p)

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--config_env', default = 'configs/env.yml',
                        help='Config file for the environment')
    # parser.add_argument('--config_exp', default = 'configs/pascal/resnet18/normals.yml',
    #                     help='Config file for the experiment')
    parser.add_argument('--config_exp', default = 'configs/kitti/hrnet32/mti_net_normals.yml',
                        help='Config file for the experiment')
    args = parser.parse_args()
    main()
