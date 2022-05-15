# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from curses import raw
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.EncoderContrastive import SupConLoss


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    positive_class_weight: int = field(
        default=1,
        metadata={
            "help": "class weight for loss function"
        },
    )


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, positive_class_weight):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.positive_class_weight = positive_class_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        if model.auto_encoder:
            ae_input, ae_hidden_output, cnn_hidden_output, ae_output, temp_net_output = net_output
            loss, outputs = self.compute_loss(model, net_output, sample, reduce=reduce)
            net_output = temp_net_output
        else:
            loss, outputs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        # print("PREDICTIONS SHAPE: ", F.softmax(net_output, dim=1).detach().cpu().numpy().shape)
        # print("PREDICTIONS: ", F.softmax(net_output, dim=1).argmax(dim=1).detach().cpu().numpy())
        # print("WHAT ABOUT THIS PREDICTIONS: ", outputs[0])
        # print("DIFFERENCES: ", F.softmax(net_output, dim=1).argmax(dim=1).detach().cpu().numpy()==outputs[0].detach().cpu().numpy())
        # print("TARGETS: ", outputs[1].detach().cpu().numpy().tolist())

        raw_predicts = F.softmax(net_output, dim=1).detach().cpu().numpy()
        if raw_predicts.shape[0] == 1:
            predicts = [raw_predicts.squeeze()[1]]
        else:
            predicts = raw_predicts[:, 1].squeeze().tolist()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "ncorrect": (outputs[0] == outputs[1]).sum(),
            "predicts": predicts,
            # "predicts": outputs[0].detach().cpu().numpy().tolist(),
            "targets": outputs[1].detach().cpu().numpy().tolist(),
        }
        # print("PREDICTION ARRAY BEFORE: ", F.softmax(net_output, dim=1))
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        if model.auto_encoder:
            ae_input, ae_hidden_output, cnn_hidden_output, ae_output, net_output = net_output

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        # print("PREDICTION SHAPE: ", lprobs.shape)
        # print("TARGET SHAPE: ", target.shape)
        # print("PREDICTION TARGET: ", target)
        # print("PREDICTIONS: ", F.softmax(net_output, dim=1).max(dim=1)[0])
        # print("DIFF: ", torch.mean((target - F.softmax(net_output, dim=1).max(dim=1)[0])**2))

        # weights = torch.tensor([3.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        # weights = weights / weights.sum()
        # weights = 1.0 / weights
        # weights = weights / weights.sum()
        # weights = weights.cuda()

        # NOTE: Log-likelihood loss
        loss = F.nll_loss(
            lprobs,
            target,
            # ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
            weight=torch.tensor([1.0, self.positive_class_weight]).to('cuda'),
            # weight=weights
        )

        if model.auto_encoder:
            ae_loss = torch.nn.MSELoss()(ae_output, ae_input)
            target_ae_hidden_output = ae_hidden_output.detach()
            # print("TARGET SHAPE: ", target_ae_hidden_output.shape)
            # print("OUTPUT SHAPE BEFORE LOG SOFTMAX: ", cnn_hidden_output.shape)
            input_cnn_hidden_output = F.log_softmax(cnn_hidden_output, dim=1)
            cl_loss = SupConLoss()(torch.stack([ae_hidden_output, cnn_hidden_output], dim=1))
            kl_loss = torch.nn.KLDivLoss(reduction="sum")(input_cnn_hidden_output, target_ae_hidden_output)
            loss = ae_loss + loss + 5e-4 * kl_loss + 5e-4 * cl_loss

        # # NOTE: Brier Score loss
        # loss = torch.mean((target - F.softmax(net_output, dim=1).max(dim=1)[0])**2)

        preds = lprobs.argmax(dim=1)
        return loss, (preds, target)

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
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
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
