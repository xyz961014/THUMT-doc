# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io

import torch
import torch.distributed as dist
import torchtext
import torchtext.data as data
from torchtext.data import Batch, Dataset

def find_path_and_exts(name1, name2):
    min_len = min(len(name1), len(name2))
    for i in range(min_len):
        if not name1[i] == name2[i]:
            break
    path = name1[:i]
    exts = (name1[i:], name2[i:])
    return path, exts

def preprocessing(string, params, add_bos=False):
    # Encode
    string = string.encode("utf-8")
    # Split string
    words = string.strip().split()
    # Append BOS and EOS
    if add_bos:
        words.insert(0, params.bos)
    else:
        words.append(params.eos)

    return words

def postprocessing(sents, params):
    sent_lens = [len(sent) for sent in sents]
    max_len = max(sent_lens)
    for i, l in enumerate(sent_lens):
        pads = [params.pad for _ in range(max_len - l)]
        sents[i] = sents[i] + pads
    return sents

def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch

def continuous_batch(data, batch_size, params):
    """Yield elements from data in chunks of batch_size."""
    max_length = params.max_length
    if hasattr(data[0], "target"):
        data = [d for d in data if len(d.source) <= max_length and len(d.target) <= max_length]
    else:
        data = [d for d in data if len(d.source) <= max_length]
    batch_len = len(data) // batch_size
    data = data[:batch_len * batch_size]
    minibatch, size_so_far = [], 0
    for sent_idx in range(batch_len):
        for batch_idx in range(batch_size):
            data_idx = batch_idx * batch_size + sent_idx
            minibatch.append(data[data_idx])
        yield minibatch
        minibatch = []
    if minibatch:
        yield minibatch

def sequence_mask(batch):
    sent_lens = [len(s) for s in batch]
    max_len = max(sent_lens)
    masks = []
    for l in sent_lens:
        mask_tensor = torch.cat((torch.ones(l), torch.zeros(max_len - l)))
        masks.append(mask_tensor.unsqueeze(0))

    return torch.cat(masks)


class TranslationDataset(Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, mode="train", **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not mode == "infer":
            src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        else:
            src_path = path

        examples = []
        if not mode == "infer":
            with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                    io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, 0, trg_line, 0, trg_line], fields))
        else:
            with io.open(src_path, mode='r', encoding='utf-8') as src_file:
                for src_line in src_file:
                    src_line = src_line.strip()
                    if src_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, 0], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class MTIterator(data.Iterator):
    
    def __init__(self, dataset, batch_size, params, mode="train", continuous=False, **kwargs):
        super().__init__(dataset, batch_size, **kwargs)
        self.continuous = continuous
        self.params = params
        self.mode = mode

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                source_batch = [m.source for m in minibatch]
                source_mask = sequence_mask(source_batch)
                if not self.mode == "infer":
                    target_batch = [m.target for m in minibatch]
                    label_batch = [m.label for m in minibatch]
                    target_mask = sequence_mask(target_batch)
                    yield Batch.fromvars(self.dataset, self.batch_size,
                                         source=postprocessing(source_batch, self.params),
                                         source_mask=source_mask,
                                         target=postprocessing(target_batch, self.params),
                                         target_mask=target_mask,
                                         label=postprocessing(label_batch, self.params))
                else:
                    yield Batch.fromvars(self.dataset, self.batch_size,
                                         source=postprocessing(source_batch, self.params),
                                         source_mask=source_mask)

            if not self.repeat:
                return

    def create_batches(self):
        if self.continuous:
            self.batches = continuous_batch(self.data(), self.batch_size, self.params)
        else:
            self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)

def build_input_fn(filenames, mode, params):
    def train_input_fn():
        path, exts = find_path_and_exts(filenames[0], filenames[1])
        SRC_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params),
                                 postprocessing=lambda s: postprocessing(s, params))
        TRG_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params, add_bos=True),
                                 postprocessing=lambda s: postprocessing(s, params))
        LABEL_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params),
                                   postprocessing=lambda s: postprocessing(s, params))
        MASK = data.Field(sequential=False, use_vocab=False)
        fields = [("source", SRC_TEXT),
                  ("source_mask", MASK),
                  ("target", TRG_TEXT),
                  ("target_mask", MASK),
                  ("label", LABEL_TEXT)]
        dataset = TranslationDataset(path, exts, fields, mode="train")

        iterator = MTIterator(dataset, params.batch_size, params, 
                              mode="train", continuous=True, sort=False, shuffle=False)

        return iterator


    def eval_input_fn():
        path, exts = find_path_and_exts(filenames[0], filenames[1])
        SRC_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params),
                                 postprocessing=lambda s: postprocessing(s, params))
        TRG_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params, add_bos=True),
                                 postprocessing=lambda s: postprocessing(s, params))
        LABEL_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params),
                                   postprocessing=lambda s: postprocessing(s, params))
        MASK = data.Field(sequential=False, use_vocab=False)
        fields = [("source", SRC_TEXT),
                  ("source_mask", MASK),
                  ("target", TRG_TEXT),
                  ("target_mask", MASK),
                  ("label", LABEL_TEXT)]
        dataset = TranslationDataset(path, exts, fields, mode="eval")

        iterator = MTIterator(dataset, params.decode_batch_size, params, 
                              mode="eval", continuous=True, sort=False, shuffle=False)

        return iterator

    def infer_input_fn():
        path, exts = filenames, []
        SRC_TEXT = data.RawField(preprocessing=lambda s: preprocessing(s, params),
                                 postprocessing=lambda s: postprocessing(s, params))
        MASK = data.Field(sequential=False, use_vocab=False)
        fields = [("source", SRC_TEXT),
                  ("source_mask", MASK)]
        dataset = TranslationDataset(path, exts, fields, mode="infer")

        iterator = MTIterator(dataset, params.decode_batch_size, params, 
                              mode="infer", continuous=True, sort=False, shuffle=False)

        return iterator

    if mode == "train":
        return train_input_fn
    if mode == "eval":
        return eval_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode %s" % mode)


def get_dataset(filenames, mode, params):
    input_fn = build_input_fn(filenames, mode, params)

    dataset = input_fn()

    return dataset

