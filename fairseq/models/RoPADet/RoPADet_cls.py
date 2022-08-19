# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from json import decoder
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.RoPADet.RoPADet import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

from fairseq.EncoderContrastive import AutoEncoder

logger = logging.getLogger(__name__)


@dataclass
class RoPADet2ClsConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to RoPADet 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside RoPADet 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside RoPADet 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside RoPADet 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune RoPADet for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in RoPADet 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in RoPADet 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded RoPADet args
    w2v_args: Any = None
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")


@dataclass
class RoPADet2CtcConfig(RoPADet2ClsConfig):
    blank_weight: float = 0
    blank_mode: str = "add"


@register_model("ropadet_ctc", dataclass=RoPADet2CtcConfig)
class RoPADetCtc(BaseFairseqModel):
    def __init__(self, cfg: RoPADet2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: RoPADet2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = RoPADetEncoder(cfg, len(task.target_dictionary))
        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@dataclass
class RoPADet2Seq2SeqConfig(RoPADet2ClsConfig):
    clf_hidden_dim: int = field(
        default=64, metadata={'help': 'classifier head hidden dimension'}
    )
    clf_dropout_rate: float = field(
        default=0.1, metadata={'help': 'classifier head dropout rate'}
    )
    clf_output_dim: int = field(
        default=2, metadata={'help': 'classifier head output dimension'}
    )
    decoder_embed_dim: int = field(
        default=256, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")
    num_classes: int = field(
        default=2,
        metadata={
            "help": "number of output classes"
        }
    )


@register_model("ropadet_seq2seq", dataclass=RoPADet2Seq2SeqConfig)
class RoPADet2Seq2SeqModel(BaseFairseqModel):
    def __init__(self, encoder, decoder, users_profile, auto_encoder, sup_contrast):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.users_profile = users_profile

        self.auto_encoder = auto_encoder
        if self.auto_encoder:
            self.ae = AutoEncoder(input_dim=192)
            self.ae2hidden = nn.Linear(8, 384)
            self.cnn2hidden = nn.Linear(384,384)
            # self.hidden = nn.Sequential(
            #     nn.Linear(256, 256), nn.ReLU(),
            #     nn.Linear(256, 256)
            # )

        self.sup_contrast = sup_contrast

    @classmethod
    def build_model(cls, cfg: RoPADet2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        # assert (
        #     cfg.autoregressive
        # ), "Please set task.autoregressive=true for seq2seq Cls models"

        # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # def build_embedding(dictionary, embed_dim):
        #     num_embeddings = len(dictionary)
        #     padding_idx = dictionary.pad()
        #     emb = Embedding(num_embeddings, embed_dim, padding_idx)
        #     return emb

        # decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        # for param in encoder.parameters():
        #     param.requires_grad = False

        decoder = cls.build_decoder(cfg, task)
        if task.cfg.profiling:
            users_profile = torch.load(task.cfg.profiles_path)
        else:
            users_profile = None

        return RoPADet2Seq2SeqModel(encoder, decoder, users_profile, task.cfg.auto_encoder, task.cfg.sup_contrast)

    @classmethod
    def build_encoder(cls, cfg: RoPADet2ClsConfig):
        return RoPADetEncoder(cfg)

    @classmethod
    def build_decoder(cls, cfg: RoPADet2Seq2SeqConfig, task: FairseqTask):
        # return TransformerDecoder(cfg, tgt_dict, embed_tokens)
        if task.cfg.profiling:
            model = torch.nn.Sequential(
                torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
                # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=cfg.clf_dropout_rate),
                torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
            )
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # torch.nn.BatchNorm1d(cfg.clf_hidden_dim),
                torch.nn.ReLU(),
                ## torch.nn.Tanh(),
                torch.nn.Dropout(p=cfg.clf_dropout_rate),
                torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim),
                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
            )
        return model

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)
        # print("ENCODER OUTPUT")
        # for k, v in encoder_out.items():
        #     print(k, type(v))

        # if encoder_out['padding_mask'] is not None:
        #     # print("ENCODER OUTPUT and PADDING MASK SHAPE: ", encoder_out['encoder_out'].shape, encoder_out['padding_mask'].shape)
        #     print("output shape: ", encoder_out['encoder_out'].shape)
        #     print("mask length: ", (1 - encoder_out['padding_mask'].long()).sum(-1))
        #     padding_mask = encoder_out['padding_mask']
        #     output_lengths = (1 - padding_mask.long()).sum(-1)
            
        #     encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'][
        #     (
        #         torch.arange(padding_mask.shape[0], device=padding_mask.device),
        #         output_lengths - 1,
        #         torch.arange(padding_mask.shape[2], device=padding_mask.device),
        #     )], dim=1)

        #     # encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'][:, :, :], dim=1)
        # else:
        #     encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)

        # NOTE: old method, with bugs when there is padding
        # because of unequal input shape
        # encoder_out['encoder_out'] = torch.mean(encoder_out['encoder_out'], dim=1)

        # NOTE: new method
        # mean pooling over time
        encoder_padding_mask = encoder_out["padding_mask"]  # B x T
        encoder_output = encoder_out['encoder_out']
        # encoder_out = preds[0].transpose(0, 1)    # B x T x C
        if encoder_padding_mask is not None:
            # preds = preds.clone()  # required because of transpose above
            encoder_output[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            encoder_out['encoder_out'] = torch.sum(encoder_output, dim=1) / ntokens.type_as(encoder_output)
        else:
            encoder_out['encoder_out'] = torch.mean(encoder_output, dim=1)

        encoder_out['encoder_out'] = encoder_out['encoder_out'].squeeze()

        if self.users_profile:
            profiles_id = kwargs['profile']
            profiles = list(map(lambda profile_id: self.users_profile[profile_id], profiles_id))
            # print("PROFILES: ", type(profiles), len(profiles))
            # for profile in profiles:
            #     # print("PROFILE: ", profile.shape)
            #     if type(profile) == list:
            #         # print(profiles_id)
            #         print(profile)
            profiles_tensor = torch.stack(profiles).to(encoder_out['encoder_out'].get_device())

            decoder_input = torch.cat((encoder_out['encoder_out'], profiles_tensor), dim=1)
            # decoder_input = encoder_out['encoder_out'] + encoder_out['encoder_out'] * F.softmax(profiles_tensor, dim=1)
        else:
            decoder_input = encoder_out['encoder_out']

        if self.auto_encoder:
            # source: (batch_size, num_mels, num_timesteps)
            # mask: (batch_size, num_mels, num_timesteps)
            x = kwargs['source']
            mask = kwargs['padding_mask']
            ae_input = torch.sum(x, dim=2)/(mask.shape[-1]-torch.sum(mask, dim=2))
            ae_bottleneck = self.ae.bottleneck(self.ae.encoder(ae_input))

            ae_bottleneck = F.normalize(ae_bottleneck, dim=1)
            decoder_input = F.normalize(decoder_input, dim=1)

            # ae_hidden_output = self.hidden(self.ae2hidden(ae_bottleneck))
            # cnn_hidden_output = self.hidden(self.cnn2hidden(decoder_input))
            # NOTE: Try without the additional hidden layers
            ae_hidden_output = self.ae2hidden(ae_bottleneck)
            cnn_hidden_output = self.cnn2hidden(decoder_input)
            ae_hidden_output = F.relu(ae_hidden_output)
            cnn_hidden_output = F.relu(cnn_hidden_output)
            ae_output = self.ae.output(self.ae.decoder(ae_bottleneck))
            return ae_input, ae_hidden_output, cnn_hidden_output, ae_output, self.decoder(decoder_input)

        if self.sup_contrast:
            return F.normalize(decoder_input, dim=1), self.decoder(decoder_input)
        # print("DECODER INPUT SHAPE: ", decoder_input.shape)

        decoder_out = self.decoder(decoder_input)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class RoPADetEncoder(FairseqEncoder):
    def __init__(self, cfg: RoPADet2ClsConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        if w2v_args.task._name == 'stft_audio_pretraining':
            model.remove_pretraining_modules()
            # print("NEW LOADING METHOD, PRINTING MODEL TYPE IN EACH STEP")
            # print("MODEL STYLE IN STEP 1: ", type(model))
            # model = model.encoder
            # print("MODEL STYLE IN STEP 2: ", type(model))
            # model = model.w2v_model
            # print("MODEL STYLE IN STEP 3: ", type(model))
        #     pass
        # else:
        # print("TYPE OF MODEL 3: ", type(model2))
        # print("TASK AND MODEL: ", w2v_args.task, w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        if w2v_args.task._name == 'audio_finetuning':
            self.w2v_model = model.encoder.w2v_model
            d = 256
        else:
            self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            # if cfg.w2v_args.task._name == 'audio_finetuning':
            #     print("NEW LOADING, MODEL TYPE: ", type(model))
            #     new_dict = {
            #         k.replace('encoder.', ''): v
            #         for (k, v) in state['model'].items()
            #         if k.startswith('encoder.')
            #     }
            # # print("MODEL STATE: ", state["model"].keys())
            # # print("MODEL: ", model)
            
            #     model.load_state_dict(new_dict, strict=True)
            # # print("NIGGA WUT")
            # else:
            model.load_state_dict(state["model"], strict=True)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # B x T x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: RoPADet2Seq2SeqConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim**-0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                    self_attn_padding_mask=self_attn_padding_mask,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
