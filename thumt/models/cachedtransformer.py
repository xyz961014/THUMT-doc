# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import thumt.utils as utils
import thumt.modules as modules

from thumt.models.transformer import AttentionSubLayer, FFNSubLayer



def compute_attention_weight(query, keys, scale=1.0):
    weights = torch.einsum("...bh,bnh->...bn", query, keys)
    weights = F.softmax(scale * weights / math.sqrt(keys.size(-1)), 2)

    return weights

def booleanize_indices(indices, cache_value):
    cache_N = len(cache_value)
    cache_k = indices.size(-1)
    batch_size = indices.size(1)

    boolean_base = torch.eye(cache_N, device=indices.device)
    indice_bool = torch.index_select(boolean_base, 0, indices.reshape(-1))
    indice_bool = indice_bool.view(-1, batch_size, cache_k, cache_N)
    indice_bool = indice_bool.sum(2)

    return indice_bool


class PositionalEmbedding(modules.Module):
    def __init__(self, d_model, name="positional_embedding"):
        super().__init__(name=name)
            
        self.d_model = d_model

        inverse_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, pos_seq):

        sinusoid = torch.einsum("bi,j->ibj", pos_seq, self.inverse_freq)

        pos_embedding = torch.cat((sinusoid.sin(), sinusoid.cos()), -1)

        return pos_embedding


class LearnableSelfAttentionSubLayer(modules.Module):

    def __init__(self, params, name="learnableselfattention"):
        super().__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.LearnableMultiHeadSelfAttention(params.hidden_size, 
                                                                     params.num_heads, 
                                                                     params.attention_dropout,
                                                                     params.enable_relative_positional_embedding)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, pos_emb, pos_bias_u, pos_bias_v, 
                cache=None, indice_bool=None, weights=None, state=None, record_kv=True):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, pos_emb, pos_bias_u, pos_bias_v, cache, indice_bool, weights)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, pos_emb, pos_bias_u, pos_bias_v, cache, indice_bool, weights, kv)
            if record_kv:
                state["k"], state["v"] = k, v

        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class Cache(modules.Module):

    def __init__(self, params, name="cache"):
        super().__init__(name=name)

        if name == "encoder_cache":
            self.update_method = params.src_update_method
            self.cache_N = params.src_cache_N
            self.cache_k = params.src_cache_k
            self.cache_dk = params.src_cache_dk
            self.summary_method = params.src_summary_method
            self.num_layers = params.num_encoder_layers
        elif name == "decoder_cache":
            self.update_method = params.tgt_update_method
            self.cache_N = params.tgt_cache_N
            self.cache_k = params.tgt_cache_k
            self.cache_dk = params.tgt_cache_dk
            self.summary_method = params.tgt_summary_method
            self.num_layers = params.num_decoder_layers
        else:
            raise ValueError("Unknown cache name %s" % name)

        self.batch_size = params.batch_size
        self.cache_L = params.max_length
        self.hidden_size = params.hidden_size

        if self.summary_method in ["sum", "max", "mean", "last_state"]:
            self.cache_dk = params.hidden_size
        elif self.summary_method == "weighted_sum":
            seld.cache_dk = params.hidden_size
            self.summary = nn.Linear(self.cache_L, 1, bias=False)
        elif self.summary_method == "conv":
            self.cache_dk = params.hidden_size
            self.num_heads = params.num_heads
            self.summary = nn.Conv1d(params.num_heads, params.num_heads, 
                                     params.hidden_size // params.num_heads, 
                                     stride=self.cache_L, 
                                     padding=math.ceil((params.hidden_size // params.num_heads \
                                                        - params.num_heads) / 2))
        elif self.summary_method == "linear":
            self.summary = nn.Sequential(nn.Linear(params.hidden_size * self.cache_L, self.cache_dk),
                                         nn.Tanh())
        else:
            raise ValueError("Unknown summary_method %s " % self.summary_method)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def new_key_and_value(self):
        cache_key = torch.zeros(self.cache_N, self.batch_size, self.cache_dk).cuda()
        #cache_value = torch.zeros(self.cache_N, 
        #                          self.batch_size, 
        #                          self.cache_L, 
        #                          1 + self.num_layers,
        #                          self.hidden_size).cuda()
        cache_value = [[torch.zeros(0, 
                                   1 + self.num_layers,
                                   self.hidden_size).cuda()
                           for _ in range(self.batch_size)]
                       for _ in range(self.cache_N)]
        return cache_key, cache_value

    def forward(self, query, keys):
        
        query = query.transpose(0, 1)
        keys = keys.transpose(0, 1)

        attention = compute_attention_weight(query, keys)

        _, topk_indices = attention.topk(self.cache_k, dim=-1)
        
        return attention, topk_indices

    def update_cache(self, key, value, hidden_state, mask):

        # key dimension:   [cache_N, batch_size, cache_dk]
        # value dimension: [cache_N, batch_size, cache_L, 1 + num_layers, hidden_size]

        if not key.size(1) == hidden_state.size(0):
            # do not update cache if batch_size does not match
            print("batch_size not match when update cache, with key/value batch_size %s and hidden batch_size %s" % 
                  (key.size(1), hidden_state.size(0)))
            return key, value

        key_blocks = list(key.split(1))
        #value_blocks = list(value.split(1))
        value_blocks = value

        # eliminate an old one
        if self.update_method == "fifo":
            key_blocks, value_blocks = self.fifo(key_blocks, value_blocks)
        
        # compute new key
        top_layer = hidden_state[:,:,-1,:]
        if self.summary_method == "sum":
            new_key = top_layer.sum(dim=1)
        elif self.summary_method == "max":
            new_key, _ = top_layer.max(dim=1)
        elif self.summary_method == "mean":
            new_key = top_layer.mean(dim=-1)
        elif self.summary_method == "last_state":
            new_key = top_layer[:,-1,:]
        elif self.summary_method == "weighted_sum":
            new_key = self.summary(top_layer.transpose(-2, -1)).squeeze(-1)
        elif self.summary_method == "conv":
            key_base = top_layer.reshape(*top_layer.size()[:2], self.num_heads, -1).transpose(1, 2)
            key_base = key_base.reshape(*key_base.size()[:2], -1)
            new_key = self.summary(key_base).reshape(key_base.size(0), -1)
        elif self.summary_method == "linear":
            new_key = self.summary(top_layer.reshape(-1, self.cache_L * self.hidden_size))

        #new_value = F.pad(hidden_state, (0, 0, 0, 0, 0, self.cache_L - hidden_state.size(-3)))
        new_value = [v.squeeze(0).detach() for v in hidden_state.split(1, dim=0)]

        key_blocks[-1] = new_key.unsqueeze(0).detach()
        value_blocks[-1] = new_value
        
        key = torch.cat(key_blocks, 0)
        #value = torch.cat(value_blocks, 0)
        value = value_blocks

        return key, value
    
    def fifo(self, key_blocks, value_blocks):

        for i in range(self.cache_N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = [torch.zeros_like(value_blocks[0][0]) for _ in range(len(value_blocks[0]))]

        return key_blocks, value_blocks


class CachedTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super().__init__(name=name)

        with utils.scope(name):
            self.self_attention = LearnableSelfAttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias, pos_emb, pos_bias_u, pos_bias_v, 
                cache=None, indice_bool=None, weights=None, record_kv=True):
        x = self.self_attention(x, bias, pos_emb, pos_bias_u, pos_bias_v, cache, indice_bool, weights, 
                                record_kv=record_kv)
        x = self.feed_forward(x)
        return x


class CachedTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super().__init__(name=name)

        with utils.scope(name):
            self.self_attention = LearnableSelfAttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                      name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, attn_bias, encdec_bias, memory, pos_emb, pos_bias_u, pos_bias_v,
                cache=None, indice_bool=None, weights=None, state=None, record_kv=True):
        x = self.self_attention(x, attn_bias, pos_emb, pos_bias_u, pos_bias_v, 
                                cache, indice_bool, weights, state=state, record_kv=record_kv)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class CachedTransformerEncoder(modules.Module):

    def __init__(self, params, name="cached_encoder"):
        super().__init__(name=name)

        self.normalization = params.normalization
        self.enable_cache = params.enable_encoder_cache
        self.enable_relative_positional_embedding = params.enable_relative_positional_embedding
        self.query_method = params.src_query_method
        self.dropout = params.residual_dropout

        with utils.scope(name):
            self.cache = Cache(params, name="encoder_cache")
            if self.query_method == "single_linear":
                self.query_transform = nn.Sequential(nn.Linear(params.hidden_size, self.cache_dk),
                                                     nn.Tanh())
            self.layers = nn.ModuleList([CachedTransformerEncoderLayer(params, name="layer_%d" % i)
                                         for i in range(params.num_encoder_layers)])
            if params.enable_relative_positional_embedding:
                self.pos_emb = PositionalEmbedding(params.hidden_size)
                self.pos_bias_u = nn.Parameter(torch.Tensor(params.num_heads, params.hidden_size // params.num_heads))
                self.pos_bias_v = nn.Parameter(torch.Tensor(params.num_heads, params.hidden_size // params.num_heads))
                self.add_name(self.pos_bias_u, "pos_bias_u")
                self.add_name(self.pos_bias_v, "pos_bias_v")
            else:
                self.pos_bias_u, self.pos_bias_v = None, None

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

        self.reset_parameters()

    def compute_pos_emb(self, x, values=None):
        if self.enable_relative_positional_embedding:
            batch_size, seq_len = x.size(0), x.size(1)
            if values is not None:
                cache_lens = [sum([value[i].size(0) for value in values]) for i in range(batch_size)]
                pos_seqs = [torch.arange(seq_len + cache_len - 1, -1, -1.0).to(x) for cache_len in cache_lens]
            else:
                pos_seqs = [torch.arange(seq_len-1, -1, -1.0).to(x) for _ in range(batch_size)]
            pos_embs = [self.pos_emb(pos_seq.unsqueeze(0)) for pos_seq in pos_seqs]
            pos_embs = [F.dropout(pos_emb, self.dropout, self.training) for pos_emb in pos_embs]
            return pos_embs
        else:
            return None

    def compute_query(self, x, bias):

        pos_emb = self.compute_pos_emb(x)

        for layer in self.layers:
            x = layer(x, bias, pos_emb, self.pos_bias_u, self.pos_bias_v, record_kv=False)

        if self.normalization == "before":
            x = self.layer_norm(x)

        if self.query_method == "single":
            query = x
        elif self.query_method == "single_linear":
            query = self.query_transform(x)
        else:
            raise ValueError("Unknown query_method %s" % self.query_method)

        return query

    def forward(self, x, bias, cache_key=None, cache_value=None): 

        if self.enable_cache:
            if x.size(0) == cache_key.size(1):
                ### compute query ###
                query = self.compute_query(x, bias)

                ### look up from cache ###
                weights, indices = self.cache(query, cache_key)

                # compute indice_bool
                indice_bool = booleanize_indices(indices, cache_value)
            else:
                indice_bool, weights, cache_value = None, None, None
        else:
            indice_bool, weights, cache_value = None, None, None

        ### compute attention ###
        pos_emb = self.compute_pos_emb(x, cache_value)
        hidden = [x.unsqueeze(-2)]

        for i, layer in enumerate(self.layers):
            if indice_bool is not None:
                value_i = [[value[:,i,:] for value in values] for values in cache_value]
            else:
                value_i = None

            x = layer(x, bias, pos_emb, self.pos_bias_u, self.pos_bias_v,
                      cache=value_i, indice_bool=indice_bool, weights=weights)
            hidden.append(x.unsqueeze(-2))

        hidden = torch.cat(hidden, dim=-2)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, hidden

    def reset_parameters(self):
        if self.enable_relative_positional_embedding:
            nn.init.constant_(self.pos_bias_u, 0.0)
            nn.init.constant_(self.pos_bias_v, 0.0)

class CachedTransformerDecoder(modules.Module):

    def __init__(self, params, name="cached_decoder"):
        super().__init__(name=name)

        self.normalization = params.normalization
        self.enable_cache = params.enable_decoder_cache
        self.enable_relative_positional_embedding = params.enable_relative_positional_embedding
        self.query_method = params.tgt_query_method
        self.dropout = params.residual_dropout

        with utils.scope(name):
            self.cache = Cache(params, name="decoder_cache")
            if self.query_method == "single_linear":
                self.query_transform = nn.Sequential(nn.Linear(params.hidden_size, self.cache_dk),
                                                     nn.Tanh())
            self.layers = nn.ModuleList([CachedTransformerDecoderLayer(params, name="layer_%d" % i)
                                         for i in range(params.num_decoder_layers)])
            if params.enable_relative_positional_embedding:
                self.pos_emb = PositionalEmbedding(params.hidden_size)
                self.pos_bias_u = nn.Parameter(torch.Tensor(params.num_heads, params.hidden_size // params.num_heads))
                self.pos_bias_v = nn.Parameter(torch.Tensor(params.num_heads, params.hidden_size // params.num_heads))
                self.add_name(self.pos_bias_u, "pos_bias_u")
                self.add_name(self.pos_bias_v, "pos_bias_v")
            else:
                self.pos_bias_u, self.pos_bias_v = None, None

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

        self.reset_parameters()

    def compute_pos_emb(self, x, values=None, k=None):
        if self.enable_relative_positional_embedding:
            batch_size, seq_len = x.size(0), x.size(1)
            if k is not None:
                k_len = k.size(1)
                seq_len += k_len
            if values is not None:
                cache_batch_size = len(values[0])
                cache_lens = [sum([value[i].size(0) for value in values]) for i in range(cache_batch_size)]
                if not cache_batch_size == batch_size:
                    beam_size = batch_size // cache_batch_size
                    cache_lens = [cache_lens[i // beam_size] for i in range(batch_size)]
                pos_seqs = [torch.arange(seq_len + cache_len - 1, -1, -1.0).to(x) for cache_len in cache_lens]
            else:
                pos_seqs = [torch.arange(seq_len-1, -1, -1.0).to(x) for _ in range(batch_size)]
            pos_embs = [self.pos_emb(pos_seq.unsqueeze(0)) for pos_seq in pos_seqs]
            pos_embs = [F.dropout(pos_emb, self.dropout, self.training) for pos_emb in pos_embs]
            return pos_embs
        else:
            return None

    def compute_query(self, x, attn_bias, encdec_bias, memory, state):

        pos_emb = self.compute_pos_emb(x)

        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory, pos_emb, self.pos_bias_u, self.pos_bias_v,
                          state=state["decoder"]["layer_%d" % i], record_kv=False)
            else:
                x = layer(x, attn_bias, encdec_bias, memory, pos_emb, self.pos_bias_u, self.pos_bias_v,
                          record_kv=False)

        if self.normalization == "before":
            x = self.layer_norm(x)

        if self.query_method == "single":
            query = x
        elif self.query_method == "single_linear":
            query = self.query_transform(x)
        else:
            raise ValueError("Unknown query_method %s" % self.query_method)

        return x

    def forward(self, x, attn_bias, encdec_bias, memory, 
                state=None, cache_key=None, cache_value=None):
        
        if self.enable_cache:
            if x.size(0) % cache_key.size(1) == 0:
                ### compute query ###
                batch_size = x.size(0)
                query = self.compute_query(x, attn_bias, encdec_bias, memory, state)

                ### look up from cache ###
                if not batch_size == self.cache.batch_size:
                    # in infer stage, query.size(0) = batch_size * beam_size 
                    query = query.reshape(self.cache.batch_size, -1, query.size(-1))

                weights, indices = self.cache(query, cache_key)

                if not batch_size == self.cache.batch_size:
                    indices = indices.transpose(0, 1).reshape(-1, batch_size, self.cache.cache_k)
                    weights = weights.transpose(0, 1).reshape(-1, batch_size, self.cache.cache_N)
                # compute indice_bool
                indice_bool = booleanize_indices(indices, cache_value)
            else:
                indice_bool, weights, cache_value = None, None, None
        else:
            indice_bool, weights, cache_value = None, None, None

        ### compute attention ###
        if not self.training and state is not None:
            k = state["decoder"]["layer_0"]["k"]
            pos_emb = self.compute_pos_emb(x, cache_value, k)
        else:
            pos_emb = self.compute_pos_emb(x, cache_value)
        hidden = [x.unsqueeze(-2)]

        for i, layer in enumerate(self.layers):
            if indice_bool is not None:
                value_i = [[value[:,i,:] for value in values] for values in cache_value]
            else:
                value_i = None

            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory, pos_emb, self.pos_bias_u, self.pos_bias_v,
                          cache=value_i,
                          indice_bool=indice_bool,
                          weights=weights,
                          state=state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, pos_emb, self.pos_bias_u, self.pos_bias_v,
                          cache=value_i,
                          indice_bool=indice_bool,
                          weights=weights)
            hidden.append(x.unsqueeze(-2))
        hidden = torch.cat(hidden, dim=-2)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, hidden

    def reset_parameters(self):
        if self.enable_relative_positional_embedding:
            nn.init.constant_(self.pos_bias_u, 0.0)
            nn.init.constant_(self.pos_bias_v, 0.0)


class CachedTransformer(modules.Module):

    def __init__(self, params, name="cachedtransformer"):
        super().__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.BatchWisePositionalEmbedding()
            self.encoder = CachedTransformerEncoder(params)
            self.decoder = CachedTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.enable_relative_positional_embedding = params.enable_relative_positional_embedding
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        self.bias = nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        src_cache_key = features["source_cache_key"]
        src_cache_value = features["source_cache_value"]
        enc_attn_bias = self.masking_bias(src_mask)

        state["source_lens"] = src_mask.sum(1).int()

        ### get embedding ###
        inputs = F.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        if not self.enable_relative_positional_embedding:
            src_starts = features["source_starts"]
            if inputs.size(0) == src_starts.size(0): 
                inputs = self.encoding(inputs, starts=src_starts)
        inputs = F.dropout(inputs, self.dropout, self.training)
            #? could consider fixed dropout 

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output, encoder_hidden = self.encoder(inputs, enc_attn_bias, src_cache_key, src_cache_value)

        state["encoder_hidden"] = encoder_hidden
        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]
        tgt_cache_key = features["target_cache_key"]
        tgt_cache_value = features["target_cache_value"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        if not mode == "infer":
            # in inference statge, the update of target_lens is in beam_search
            state["target_lens"] = features["target_mask"].sum(1).int()

        ### get embedding ###
        targets = F.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)
        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        if not self.enable_relative_positional_embedding:
            tgt_starts = features["target_starts"]
            if decoder_input.size(0) % tgt_starts.size(0) == 0:
                if mode == "infer":
                    tgt_starts = tgt_starts.expand(decoder_input.size(0) // tgt_starts.size(0), -1).reshape(-1)
                if decoder_input.size(0) == tgt_starts.size(0): 
                    decoder_input = self.encoding(decoder_input, starts=tgt_starts)
        decoder_input = F.dropout(decoder_input, self.dropout, self.training)
            #? could consider fixed dropout 

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output, decoder_hidden = self.decoder(decoder_input, dec_attn_bias,
                                                      enc_attn_bias, encoder_output, state,
                                                      tgt_cache_key, tgt_cache_value)
        state["decoder_hidden"] = decoder_hidden

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)


        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0], labels.device)
        state = self.encode(features, state)
        logits, state = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(logits)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return  torch.exp(-loss) * mask - (1 - mask)

        return torch.sum(loss * mask) / torch.sum(mask), state

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                } for i in range(self.num_decoder_layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # cache params
            src_cache_N=5,
            src_cache_k=2,
            src_cache_dk=512,
            tgt_cache_N=5,
            tgt_cache_k=2,
            tgt_cache_dk=512,
            src_query_method="single",
            src_summary_method="last_state",
            src_update_method="fifo",
            tgt_query_method="single",
            tgt_summary_method="last_state",
            tgt_update_method="fifo",
            enable_encoder_cache=True,
            enable_decoder_cache=True,
            enable_relative_positional_embedding=True,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def test_params():
        params = CachedTransformer.base_params()
        params.hidden_size = 64
        params.filter_size = 256
        params.num_heads = 4
        params.residual_dropout = 0.0
        params.learning_rate = 5e-4
        params.train_steps = 100000
        params.num_encoder_layers = 3
        params.num_decoder_layers = 3

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return CachedTransformer.base_params()
        elif name == "test":
            return CachedTransformer.test_params()
        else:
            return CachedTransformer.base_params()
