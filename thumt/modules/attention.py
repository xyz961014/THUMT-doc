# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.o_transform = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

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
                k = torch.cat([kv[0].to(k), k], dim=1)
                v = torch.cat([kv[1].to(v), v], dim=1)

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

        weights = self.dropout(torch.softmax(logits, dim=-1))

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_transform.weight)
        nn.init.xavier_uniform_(self.k_transform.weight)
        nn.init.xavier_uniform_(self.v_transform.weight)
        nn.init.xavier_uniform_(self.o_transform.weight)
        nn.init.constant_(self.q_transform.bias, 0.0)
        nn.init.constant_(self.k_transform.bias, 0.0)
        nn.init.constant_(self.v_transform.bias, 0.0)
        nn.init.constant_(self.o_transform.bias, 0.0)

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