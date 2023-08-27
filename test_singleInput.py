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
from evaluation.evaluate_utils_test_singleInp import eval_model, validate_results, save_model_predictions,\
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
    
    

    
    print(colored('Starting Final Predictions', 'green'))
    
    checkpoint = torch.load(p['checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
        
    save_model_predictions(p, args.singleInput_path, args.img_size, model)
    

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--config_env', default = 'configs/env.yml',
                        help='Config file for the environment')
    # parser.add_argument('--config_exp', default = 'configs/pascal/resnet18/normals.yml',
    #                     help='Config file for the experiment')
    parser.add_argument('--config_exp', default = 'configs/kitti/hrnet32/mti_net_normals.yml',
                        help='Config file for the experiment')
    parser.add_argument('--singleInput_path', default = '/path/to/databases/KITTIRoad/training/',
                        help='Config file for the experiment')
    parser.add_argument('--img_size', default = (1280, 384),
                        help='Config file for the experiment')
    args = parser.parse_args()
    main()
