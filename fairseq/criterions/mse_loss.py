# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from curses import raw
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.EncoderContrastive import SupConLoss
from typing import List
import numpy as np

@dataclass
class MSECriterionConfig(FairseqDataclass):
    l2_loss: bool = field(
        default=True,
        metadata={
            "help": "whether to use l2 or l1 loss"
        },
    )


@register_criterion("mse_loss", dataclass=MSECriterionConfig)
class MSECriterion(FairseqCriterion):
    def __init__(self, task, l2_loss):
        super().__init__(task)
        self.l2_loss = l2_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample["net_input"], features_only=True)
        expected = {key: sample["net_input"][key] for key in model.forward.__code__.co_varnames if key in sample["net_input"].keys()}
        net_output = model(**expected, features_only=True)
        loss, outputs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0)
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "ncorrect": (outputs[0] == outputs[1]).sum(),
            "predicts": net_output["x"],
            "targets": sample["target"],
        }
        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        preds = net_output["x"]
        # mean pooling over time
        encoder_padding_mask = net_output["padding_mask"]  # B x T
        # encoder_out = preds[0].transpose(0, 1)    # B x T x C
        if encoder_padding_mask is not None:
            preds = preds.clone()  # required because of transpose above
            preds[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(preds, dim=1) / ntokens.type_as(preds)
        else:
            x = torch.mean(preds, dim=1)

        x = x.squeeze()

        targets = sample["target"]
        # print("PREDICTION: ", preds.shape, preds.device)
        # print("MASK at input 0: ", net_output["padding_mask"][0,:], len(net_output["padding_mask"][0,:]))
        # print("MASK at input -1: ", net_output["padding_mask"][-1,:], len(net_output["padding_mask"][-1,:]))
        # print("TARGET: ", targets.shape, targets.device)
        if self.l2_loss:
            loss = F.mse_loss(
                            x,
                            targets,
                            reduction="sum" if reduce else "none"
                            )
        else:
            loss = F.l1_loss(
                             x,
                             targets,
                             reduction="sum" if reduce else "none"
                            )

        return loss.sum(), (x, targets)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "mse_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["mse_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar(
            "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=3
        )

        # predicts = [log.get("predicts") for log in logging_outputs]
        # targets = [log.get("targets") for log in logging_outputs]

        # flat_predicts = [item for predict in predicts for item in predict]
        # flat_targets = [item for target in targets for item in target]
        # print("TARGET: ", flat_targets)
        # print("PREDICTIONS: ", flat_predicts)
        # metrics.log_scalar(
        #     "auc", roc_auc_score(flat_targets, flat_predicts), round=3
        # )
        # metrics.log_derived()


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
