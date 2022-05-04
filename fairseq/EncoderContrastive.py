from torch.nn.modules.utils import _pair, _quadruple
from collections import OrderedDict
from torch.nn import init
from torch import Tensor, einsum
from typing import Any, Callable, List, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self,
                 num_layer=4,
                 input_dim=640,
                 hidden_dim=128,
                 bottleneck_dim=8,
                 **kwargs):
        super(AutoEncoder, self).__init__()
        input_d = input_dim
        modules = []
        for i in range(num_layer):
            modules.append(
                nn.Sequential(nn.Linear(input_d, hidden_dim),
                              nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(inplace=True)))
            input_d = hidden_dim
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Sequential(nn.Linear(input_d, bottleneck_dim),
                                        nn.BatchNorm1d(bottleneck_dim),
                                        nn.ReLU(inplace=True))
        input_d = bottleneck_dim
        modules = []
        for i in range(num_layer):
            modules.append(
                nn.Sequential(nn.Linear(input_d, hidden_dim),
                              nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(inplace=True)))
            input_d = hidden_dim
        self.decoder = nn.Sequential(*modules)
        self.output = nn.Linear(input_d, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).flatten(1, -1)
        feature = x.detach()
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.output(x)
        return x, feature

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,
                 temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
