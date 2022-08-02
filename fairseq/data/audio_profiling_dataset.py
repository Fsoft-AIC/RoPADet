# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset, data_utils


class AudioProfilingDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        targets,
        batch_targets,
        process_target = None,
    ):
        super().__init__(dataset)
        self.targets = targets
        self.batch_targets = batch_targets
        self.process_target = process_target

    def get_target(self, index, process_fn=None):
        profile_id = self.dataset[index]['profile']
        target = self.targets[profile_id]
        return target if process_fn is None else process_fn(target)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["target"] = self.get_target(index, process_fn=self.process_target)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_target(index))
        return sz, own_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = collated["id"].tolist()
        targets = [s["target"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in targets])
            # target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            size = max(t.size(0) for t in targets)
            batch_size = len(targets)
            # print(batch_size)
            # print(size)
            # print(targets[0])
            # print(targets[0].device)
            final_target = targets[0].new(batch_size, size).fill_(0.0)
            for i, t in enumerate(targets):
                assert final_target[i].numel() == t.numel()
                final_target[i].copy_(t)

            # print("DEVICE: ", target[0].device)
            # for t in target:
            #     print(t.shape, t.device, t.type())
            # print(torch.stack(target, dim=0))
            # final_target = torch.stack(targets, dim=0) # shape: Batch x Dim
            # final_target = targets
            collated["ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in targets])

        collated["target"] = final_target

        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
