import torch.nn as nn
import torch.nn.functional as F

def compute_seg_loss(pred, target):
    """计算分割损失"""
    return F.cross_entropy(pred, target)

def compute_cls_loss(pred, target):
    """计算分类损失"""
    return F.cross_entropy(pred, target) 