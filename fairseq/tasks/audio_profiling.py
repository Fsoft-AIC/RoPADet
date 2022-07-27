# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch
import json

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from fairseq.data import AddTargetDataset, Dictionary, encoders
from fairseq.data.audio_profiling_dataset import AudioProfilingDataset
from fairseq.tasks.stft_audio_pretraining import STFTAudioPretrainingConfig, STFTAudioPretrainingTask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from . import register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        # print(self.dictionary.encode_line(
        #     label, append_eos=False, add_if_not_exist=False
        # ).type(torch.LongTensor))
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        ).type(torch.LongTensor) - 4


@register_task("audio_profiling", dataclass=STFTAudioPretrainingConfig)
class AudioProfilingTask(STFTAudioPretrainingTask):
    """ """

    cfg: STFTAudioPretrainingConfig

    def __init__(
        self,
        cfg: STFTAudioPretrainingConfig,
    ):
        super().__init__(cfg)


    def load_dataset(
        self, split: str, task_cfg: STFTAudioPretrainingConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        # assert task_cfg.labels is not None
        # text_compression_level = getattr(
        #     TextCompressionLevel, str(self.cfg.text_compression_level)
        # )
        # data_path = self.cfg.data
        profiles_path = self.cfg.profiles_path
        profiles = torch.load(profiles_path)
        for key, value in profiles.items():
            profiles[key] = value.cpu()
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        # text_compressor = TextCompressor(level=text_compression_level)

        # labels = []
        # with open(label_path, "r") as f:
        #     # for line in f:
        #     #     label = line.strip()
        #     #     labels.append(torch.LongTensor([self.label_vocab.add_symbol(label)]))
        #     labels = [
        #         text_compressor.compress(l.strip())
        #         for i, l in enumerate(f)
        #         if i not in skipped_indices
        #     ]

        # assert len(labels) == len(self.datasets[split]), (
        #     f"labels length ({len(labels)}) and dataset length "
        #     f"({len(self.datasets[split])}) do not match"
        # )

        # process_label = LabelEncoder(self.target_dictionary)

        self.datasets[split] = AudioProfilingDataset(
            self.datasets[split],
            profiles,
            batch_targets=True,
        )

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.label_vocab

    # def valid_step(self, sample, model, criterion):
    #     loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
    #     if self.cfg.eval_wer and self.cfg.autoregressive:
    #         metrics = self._inference_with_wer(self.sequence_generator, sample, model)
    #         logging_output["_num_char_errors"] = metrics["num_char_errors"]
    #         logging_output["_num_chars"] = metrics["num_chars"]
    #         logging_output["_num_word_errors"] = metrics["num_word_errors"]
    #         logging_output["_num_words"] = metrics["num_words"]
    #     if self.cfg.eval_bleu and self.cfg.autoregressive:
    #         metrics = self._inference_with_bleu(self.sequence_generator, sample, model)
    #         logging_output["_bleu_sys_len"] = metrics.sys_len
    #         logging_output["_bleu_ref_len"] = metrics.ref_len
    #         # we split counts into separate entries so that they can be
    #         # summed efficiently across workers using fast-stat-sync
    #         assert len(metrics.counts) == 4
    #         for i in range(4):
    #             logging_output[f"_bleu_counts_{i}"] = metrics.counts[i]
    #             logging_output[f"_bleu_totals_{i}"] = metrics.totals[i]
    #     return loss, sample_size, logging_output

    # def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
    #     model = super().build_model(model_cfg, from_checkpoint)

    #     if self.cfg.eval_wer and self.cfg.autoregressive:
    #         self.sequence_generator = self.build_generator(
    #             [model],
    #             self.cfg.eval_wer_config,
    #         )
    #         if self.cfg.eval_wer_tokenizer:
    #             self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
    #         else:
    #             self.tokenizer = None
    #     if self.cfg.eval_bleu and self.cfg.autoregressive:
    #         assert self.cfg.eval_bleu_detok is not None, (
    #             "--eval-bleu-detok is required if using --eval-bleu; "
    #             "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
    #             "to disable detokenization, e.g., when using sentencepiece)"
    #         )
    #         detok_args = json.loads(self.cfg.eval_bleu_detok_args)
    #         self.tokenizer = encoders.build_tokenizer(
    #             Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
    #         )
    #         gen_args = json.loads(self.cfg.eval_bleu_args)
    #         gen_args = Namespace(**gen_args)
    #         self.sequence_generator = self.build_generator([model], gen_args)

    #     return model
