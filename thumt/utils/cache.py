# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def update_cache(model, features, state, last_feature, evaluate=False):
    if state is not None:
        src_key, src_value, src_mask = model.encoder.cache.update_cache(last_feature["source_cache_key"], 
                                                                        last_feature["source_cache_value"],
                                                                        last_feature["source_cache_mask"],
                                                                        state["encoder_hidden"],
                                                                        state["source_mask"])
        tgt_key, tgt_value, tgt_mask = model.decoder.cache.update_cache(last_feature["target_cache_key"],
                                                                        last_feature["target_cache_value"],
                                                                        last_feature["target_cache_mask"],
                                                                        state["decoder_hidden"],
                                                                        state["target_mask"])
    else:
        src_key, src_value, src_mask = model.encoder.cache.new_key_and_value()
        tgt_key, tgt_value, tgt_mask = model.decoder.cache.new_key_and_value()


    if not evaluate:
        features[0]["source_cache_key"] = src_key
        features[0]["source_cache_value"] = src_value
        features[0]["source_cache_mask"] = src_mask
        features[0]["target_cache_key"] = tgt_key
        features[0]["target_cache_value"] = tgt_value
        features[0]["target_cache_mask"] = tgt_mask
    else:
        features["source_cache_key"] = src_key
        features["source_cache_value"] = src_value
        features["source_cache_mask"] = src_mask
        features["target_cache_key"] = tgt_key
        features["target_cache_value"] = tgt_value
        features["target_cache_mask"] = tgt_mask

    return features

def update_starts(params, features, state, last_feature, evaluate=False):
    if state is not None and "source_lens" in state.keys() and "target_lens" in state.keys():
        # update starts position
        if last_feature["source_starts"].size(0) == state["source_lens"].size(0):
            src_starts = last_feature["source_starts"] + state["source_lens"]
        else:
            src_starts = last_feature["source_starts"]

        if last_feature["target_starts"].size(0) == state["target_lens"].size(0):
            tgt_starts = last_feature["target_starts"] + state["target_lens"]
        else:
            tgt_starts = last_feature["target_starts"]
    else:
        # init starts position
        if not evaluate:
            src_starts = features[0]["source"].new_zeros(features[0]["source"].size(0))
            tgt_starts = features[0]["target"].new_zeros(features[0]["target"].size(0))
        else:
            src_starts = features["source"].new_zeros(features["source"].size(0), dtype=torch.int)
            tgt_starts = features["source"].new_zeros(features["source"].size(0), dtype=torch.int)

    if not evaluate:
        features[0]["source_starts"] = src_starts
        features[0]["target_starts"] = tgt_starts
    else:
        features["source_starts"] = src_starts
        features["target_starts"] = tgt_starts

    return features




