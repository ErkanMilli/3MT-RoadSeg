#
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import torch
import torch.nn.functional as F
import numpy as np

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)
    
    if task == 'normals':
        # output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
        pass
    
    elif task in {'semseg', 'human_parts'}:
        # _, output = torch.max(output, dim=3)
        output = output[:,:,:,1]
    
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output


def get_output_val(output, task):
    output = output.permute(0, 2, 3, 1)
    
    if task == 'normals':
        # output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
        pass
    
    elif task in {'semseg', 'human_parts'}:
        # # _, output = torch.max(output, dim=3)
        output = F.softmax(output, dim=3)
        # # # output = F.relu(output)
        output = output[:,:,:,1]
        # _, output = torch.max(output, dim=3)
        
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]




def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=500, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 0.1
    return lr


def fast_hist(a, b, n):
    '''
	a and b are predict and mask respectively
	n is the number of classes
	'''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        F_score = 2 * (recall * pre) / (recall + pre)
        fpr = conf_matrix[0, 1] / np.float(conf_matrix[0, 0] + conf_matrix[0, 1])
        fnr = conf_matrix[1, 0] / np.float(conf_matrix[1, 0] + conf_matrix[1, 1])
    return F_score, pre, recall, fpr, fnr