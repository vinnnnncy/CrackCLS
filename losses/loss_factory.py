from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np

device = None


def get_loss(config):
    f = globals().get(config.loss.name)
    global device
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return f(**config.loss.params)


def l1loss(reduction='mean', **_):
    l1loss_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(pred_dict, HR, **_):
        gt_loss = 0
        loss_dict = dict()
        pred_hr = pred_dict['hr']
        gt_loss = l1loss_fn(pred_hr, HR)

        loss_dict['loss'] = gt_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':l1loss_fn}

def vid_loss(reduction='mean', lambda1=1, lambda2=1, lambda3=1, lambda4=1, T=2, epsilon=1e-8,
             pdf='gaussian', **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    gt_loss_fn = l1loss_fn
    clsloss_fn = nn.CrossEntropyLoss()

    def vid_loss_fn(mu, std, tl):
        if pdf == 'laplace':
            std = std * 0.1 + epsilon
            numerator = torch.abs(mu - tl)
            loss = mu.shape[1] * np.log(2*math.pi)/2 + torch.log(2*std) + numerator / (std)
        elif pdf == 'gaussian':
            std = std * 0.001 + epsilon
            numerator = (mu - tl) ** 2
            loss = mu.shape[1] * np.log(2*math.pi)/2 + torch.log(std)/2 + numerator / (2 * std)
        
        loss = loss.mean()
        return loss


    def loss_fn(teacher_pred_dict, student_pred_dict, HR, y ):
        gt_loss = 0
        distill_loss = 0
        # clsloss_fn = nn.CrossEntropyLoss()
        cls_distillloss_fn = nn.KLDivLoss()

        loss_dict = dict()
        student_pred_hr = student_pred_dict['hr']
        teacher_pred_hr = teacher_pred_dict['hr']
        student_pred_y = student_pred_dict['s_y']
        teacher_pred_y = teacher_pred_dict['t_y']

        for k, v in student_pred_dict.items():
            if 'mean' in k and 'sub' not in k and 'add' not in k:
                layer_name = k.split('_mean')[0]
                tl = teacher_pred_dict[layer_name]
                mu = student_pred_dict['%s_mean'%layer_name]
                std = student_pred_dict['%s_var'%layer_name]
                distill_loss += vid_loss_fn(mu, std, tl)

        gt_loss = gt_loss_fn(student_pred_hr, teacher_pred_hr)
        cls_loss = clsloss_fn(student_pred_y, y)
        cls_distill_loss = cls_distillloss_fn(F.log_softmax(student_pred_y/T, dim=1),
                             F.softmax(teacher_pred_y/T, dim=1)) * (T * T)

        
        loss_dict['gt_loss'] = lambda1 * gt_loss
        loss_dict['distill_loss'] = lambda2 * distill_loss
        loss_dict['cls_loss'] = lambda3 * cls_loss
        loss_dict['cls_distill_loss'] = lambda4 * cls_distill_loss
        loss_dict['loss'] = loss_dict['gt_loss'] + loss_dict['distill_loss'] + loss_dict['cls_loss'] + loss_dict['cls_distill_loss']

        return loss_dict

    return {'train':loss_fn,
            'val':clsloss_fn}


def teacher_LR_constraint_loss(reduction='mean',
                      lambda1=1, lambda2=1, **_):

    l1loss_fn = torch.nn.L1Loss(reduction=reduction)
    gt_loss_fn = l1loss_fn
    imitation_loss_fn = l1loss_fn
    clsloss_fn = nn.CrossEntropyLoss()


    def loss_fn(pred_dict, LR, HR, y):
        gt_loss = 0
        cls_loss = 0

        loss_dict = dict()
        pred_hr = pred_dict['hr']
        pred_y = pred_dict['t_y']

        gt_loss = gt_loss_fn(pred_hr, HR)
        cls_loss = clsloss_fn(pred_y, y)

        loss_dict['gt_loss'] = lambda1 * gt_loss
        loss_dict['cls_loss'] = lambda2 * cls_loss
        loss_dict['loss'] = loss_dict['gt_loss'] + loss_dict['cls_loss']

        return loss_dict

    return {'train':loss_fn,
            'val':clsloss_fn}
