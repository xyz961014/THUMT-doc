# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def update_cache(model, features, state, last_feature, evaluate=False):
    if state is not None:
        src_key, src_value = model.encoder.cache.update_cache(last_feature["source_cache_key"], 
                                                              last_feature["source_cache_value"],
                                                              state["encoder_hidden"])
        tgt_key, tgt_value = model.decoder.cache.update_cache(last_feature["target_cache_key"],
                                                              last_feature["target_cache_value"],
                                                              state["decoder_hidden"])
    else:
        src_key, src_value = model.encoder.cache.new_key_and_value()
        tgt_key, tgt_value = model.decoder.cache.new_key_and_value()


    if not evaluate:
        features[0]["source_cache_key"] = src_key
        features[0]["source_cache_value"] = src_value
        features[0]["target_cache_key"] = tgt_key
        features[0]["target_cache_value"] = tgt_value
    else:
        features["source_cache_key"] = src_key
        features["source_cache_value"] = src_value
        features["target_cache_key"] = tgt_key
        features["target_cache_value"] = tgt_value

    return features

def update_starts(params, features, state, last_feature, evaluate=False):
    if state is not None:
        # update starts position
        src_starts = last_feature["source_starts"] + state["source_starts"]
        tgt_starts = last_feature["target_starts"] + state["target_starts"]
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




