# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def update_cache(model, features, state, last_feature, evaluate=False):
    if state is not None:
        src_key, src_value = model.encoder.cache.update_cache(last_feature["source_cache_key"], 
                                                              last_feature["source_cache_value"],
                                                              state["encoder_output"])
        tgt_key, tgt_value = model.decoder.cache.update_cache(last_feature["target_cache_key"],
                                                              last_feature["target_cache_value"],
                                                              state["decoder_output"])
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



