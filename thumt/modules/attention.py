# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import thumt.utils as utils

from thumt.modules.module import Module
from thumt.modules.affine import Affine


class Attention(Module):

    def __init__(self, q_size, k_size, hidden_size, name="attention"):
        super(Attention, self).__init__(name)

        self._q_size = q_size
        self._k_size = k_size
        self._hidden_size = hidden_size

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size, name="q_transform")
            self.k_transform = Affine(k_size, hidden_size, name="k_transform")
            self.v_transform = Affine(hidden_size, 1,
                                      name="v_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, 1]
        logits = torch.transpose(logits, 1, 2)
        # [batch, 1, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.softmax(logits, dim=-1)

        # [batch, 1, length]
        weights = torch.squeeze(weights, 2)
        output = torch.matmul(weights, memory)

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight)
            nn.init.xavier_uniform_(self.k_transform.weight)
            nn.init.xavier_uniform_(self.v_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAttentionBase(Module):

    def __init__(self, name="multihead_attention_base"):
        super(MultiHeadAttentionBase, self).__init__(name=name)

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")

        self.reset_parameters()

    def forward(self, query, bias, memory=None, kv=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAdditiveAttention(MultiHeadAttentionBase):

    def __init__(self, q_size, k_size, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAdditiveAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(k_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, num_heads,
                                      name="v_transform")
            self.o_transform = Affine(k_size, k_size,
                                      name="o_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, num_heads]
        logits = torch.transpose(logits, 1, 2)
        # [batch, num_heads, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        vh = self.split_heads(memory, self.num_heads)
        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class PositionalEmbedding(Module):
    def __init__(self, d_model, name="positional_embedding"):
        super().__init__(name=name)
            
        self.d_model = d_model

        self.inverse_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))

    def forward(self, pos_seq):

        sinusoid = torch.einsum("bi,j->ibj", pos_seq, self.inverse_freq)

        pos_embedding = torch.cat((sinusoid.sin(), sinusoid.cos()), -1)

        return pos_embedding



class LearnableMultiHeadSelfAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0, enable_rel_emb=True, enable_sent_emb=False, gated=False,
                 name="learnable_multihead_selfattention"):
        super().__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.enable_rel_emb = enable_rel_emb
        self.enable_sent_emb = enable_sent_emb
        self.gated = gated

        if enable_sent_emb:
            self.sent_emb = PositionalEmbedding(hidden_size)

        if gated:
            self.W_c = Affine(hidden_size, hidden_size, name="W_c")
            self.W_i = Affine(hidden_size, hidden_size, name="W_i")

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")
            if self.enable_rel_emb:
                self.r_transform = Affine(hidden_size, hidden_size,
                                          name="r_transform")

        self.reset_parameters()

    def _rel_shift(self, x):
        x = x.permute(2, 3, 0, 1)
        x_inp = x.reshape(x.size(0), -1, *x.size()[-2:])
        zero_pad = x_inp.new_zeros((x_inp.size(0), 1, *x_inp.size()[2:]))
        x_padded = torch.cat([zero_pad, x_inp], dim=1)

        x_padded = x_padded.view(x_inp.size(1) + 1, x_inp.size(0), *x_inp.size()[2:])

        x = x_padded[1:].view_as(x)
        x = x.permute(2, 3, 0, 1)

        return x

    def forward(self, x, bias, pos_embs, pos_bias_u, pos_bias_v, 
                cache=None, cache_mask=None, indice_bool=None, sent_weights=None, kv=None):

        # x: [batch_size, length, hidden_size]
        # in inference stage, the batch_size here is actually batch_size * beam_size
        batch_size, x_len, hidden_size = x.size()
        head_size = hidden_size // self.num_heads

        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)
        if self.enable_rel_emb:
            rs = [self.r_transform(pos_emb.transpose(0, 1)) for pos_emb in pos_embs]

        if kv is not None:
            k = torch.cat([kv[0], k], dim=1)
            v = torch.cat([kv[1], v], dim=1)

        k_len = k.size(1)

        if cache is not None:
            cache_N = len(cache) 
            cache_batch_size = cache[0].size(0)
            cache_lens = [value.size(1) for value in cache]
            cache_L = max(cache_lens)
            cache_len = cache_L * cache_N
            beam_size = batch_size // cache_batch_size

            # append pad to the end to form same length block
            cache_tensor = torch.cat([F.pad(value, (0, 0, 0, cache_L - value.size(1))).unsqueeze(0) 
                                      for value in cache],
                                     dim=0)
            if self.enable_sent_emb:
                sent_emb = self.sent_emb(torch.arange(cache_N).unsqueeze(0)).to(cache_tensor).unsqueeze(1)
                cache_tensor = cache_tensor + sent_emb
            cache_tensor = cache_tensor.transpose(0, 1).reshape(cache_batch_size, -1, hidden_size)
            cache_k = self.k_transform(cache_tensor)
            cache_v = self.v_transform(cache_tensor)

            cache_kh = self.split_heads(cache_k, self.num_heads)
            cache_vh = self.split_heads(cache_v, self.num_heads)
            cache_kh = cache_kh.reshape(cache_batch_size, self.num_heads, cache_N, cache_L, head_size)
            cache_vh = cache_vh.reshape(cache_batch_size, self.num_heads, cache_N, cache_L, head_size)
            cache_kh = cache_kh.permute(2, 0, 1, 3, 4)
            cache_vh = cache_vh.permute(2, 0, 1, 3, 4)
            if not cache_batch_size == batch_size:
                # in inference stage, expand cache to fit beam_size
                cache_kh = cache_kh.unsqueeze(2).expand(-1, -1, beam_size, -1, -1, -1)
                cache_kh = cache_kh.reshape(cache_N, batch_size, self.num_heads, cache_L, head_size)
                cache_vh = cache_vh.unsqueeze(2).expand(-1, -1, beam_size, -1, -1, -1)
                cache_vh = cache_vh.reshape(cache_N, batch_size, self.num_heads, cache_L, head_size)
        else:
            cache_len = 0        

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)
        if self.enable_rel_emb:
            rhs = [self.split_heads(r, self.num_heads) for r in rs]

        if cache is not None and self.enable_rel_emb:
            rh = torch.cat(rhs, dim=0)
            cache_rh, rh = rh.split([cache_len, k_len], dim=2)
            cache_rhs = cache_rh.split(cache_lens, dim=2)
            cache_rh = torch.cat([F.pad(pos, (0, 0, 0, cache_L - pos.size(2))) for pos in cache_rhs], dim=2)
            cache_rh = cache_rh.reshape(batch_size, self.num_heads, cache_N, cache_L, head_size)
            cache_rh = cache_rh.permute(2, 0, 1, 3, 4)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5


        if self.enable_rel_emb:
            quh = qh + pos_bias_u[:,None,:]
            qvh = qh + pos_bias_v[:,None,:]
        else:
            quh = qh
            qvh = qh

        # dot-product attention
        kh = kh.transpose(-2, -1)
        AC = torch.matmul(quh, kh)
        if self.enable_rel_emb:
            rh = torch.cat(rhs, dim=0)
            rh = rh.transpose(-2, -1)
            BD = torch.matmul(qvh, rh)

        if bias is not None:
            AC = AC + bias

        if indice_bool is not None:
            pre_AC = torch.einsum("bnid,ibk->kbnid", quh, indice_bool)
            cache_kh = cache_kh.transpose(-2, -1)
            cache_AC = torch.matmul(pre_AC, cache_kh)
            if cache_mask is not None:
                cache_masks = [F.pad(m, (0, cache_L - m.size(-1))).unsqueeze(0) for m in cache_mask]
                cache_mask = torch.cat(cache_masks, dim=0)
                if not cache_batch_size == batch_size:
                    # in inference stage, expand cache to fit beam_size
                    cache_mask = cache_mask.unsqueeze(2).expand(-1, -1, beam_size, -1)
                    cache_mask = cache_mask.reshape(cache_N, batch_size, cache_L)
                cache_bias = (1.0 - cache_mask.float()) * -1e9
                cache_bias = cache_bias[:,:,None,None,:]
                cache_AC = cache_AC + cache_bias
            cache_AC = cache_AC.permute(1, 2, 3, 0, 4)
            cache_AC = cache_AC.reshape(*cache_AC.size()[:3], -1)
            if not self.gated:
                AC = torch.cat((cache_AC, AC), dim=-1)

            if self.enable_rel_emb:
                pre_BD = torch.einsum("bnid,ibk->kbnid", qvh, indice_bool)
                cache_rh = cache_rh.transpose(-2, -1)
                cache_BD = torch.matmul(pre_BD, cache_rh)
                cache_BD = cache_BD.permute(1, 2, 3, 0, 4)
                cache_BD = cache_BD.reshape(*cache_BD.size()[:3], -1)
                BD = torch.cat((cache_BD, BD), dim=-1)

        if self.enable_rel_emb:
            BD = self._rel_shift(BD)
            logits = AC + BD
        else:
            logits = AC

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)
        if self.gated and cache_len > 0:
            cache_weights = F.dropout(torch.softmax(cache_AC, dim=-1),
                                      p=self.dropout,
                                      training=self.training)

        if cache_len > 0:
            if not self.gated:
                cache_weights, weights = weights.split([cache_L * cache_N, k_len], dim=-1)
            if sent_weights is not None:
                cache_weights = cache_weights.reshape(*cache_weights.size()[:-1], cache_N, cache_L)
                cache_weights = cache_weights.permute(3, 0, 1, 2, 4)
                cache_weights = torch.einsum("kbnij,ibk->bnikj", cache_weights, sent_weights)
                cache_weights = cache_weights.reshape(*cache_weights.size()[:-2], -1)

            #cache_weights = cache_weights.split(cache_L, dim=-1)
            #cache_weights = torch.cat([cache_weights[i].narrow(-1, 0, cache_lens[i]) for i in range(cache_N)], dim=-1)

            #cache_vhs = cache_vh.split(1, dim=0)
            #cache_vh = torch.cat([cache_vhs[i].narrow(-2, 0, cache_lens[i]).squeeze(0) 
            #                      for i in range(cache_N)], 
            #                     dim=-2)
            cache_vh = cache_vh.reshape(batch_size, self.num_heads, cache_len, head_size)
            cache_output = torch.matmul(cache_weights, cache_vh)

        x = torch.matmul(weights, vh)

        if cache_len > 0:
            if not self.gated:
                x = x + cache_output
            else:
                lamb = torch.sigmoid(self.W_i(x) + self.W_c(cache_output))
                x = lamb * x + (1 - lamb) * cache_output

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
            if self.enable_rel_emb:
                nn.init.xavier_uniform_(self.r_transform.weight, 2 ** -0.5)
                nn.init.constant_(self.r_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


