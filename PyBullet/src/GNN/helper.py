import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase

class LayerNormGRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

        self.ln_resetgate = nn.LayerNorm(hidden_size)
        self.ln_inputgate = nn.LayerNorm(hidden_size)
        self.ln_newgate = nn.LayerNorm(hidden_size)
        self.ln = {
            'resetgate': self.ln_resetgate,
            'inputgate': self.ln_inputgate,
            'newgate': self.ln_newgate,
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)
        return self._LayerNormGRUCell(
            input, hx,
            self.weight_ih, self.weight_hh, self.ln,
            self.bias_ih, self.bias_hh,
        )

    def _LayerNormGRUCell(self, input, hidden, w_ih, w_hh, ln, b_ih=None, b_hh=None):
    	
	    gi = F.linear(input, w_ih, b_ih)
	    gh = F.linear(hidden, w_hh, b_hh)
	    i_r, i_i, i_n = gi.chunk(3, 1)
	    h_r, h_i, h_n = gh.chunk(3, 1)

	    # use layernorm here
	    resetgate = torch.sigmoid(ln['resetgate'](i_r + h_r))
	    inputgate = torch.sigmoid(ln['inputgate'](i_i + h_i))
	    newgate = torch.tanh(ln['newgate'](i_n + resetgate * h_n))
	    hy = newgate + inputgate * (hidden - newgate)

	    return hy

class fc_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn):
        super(fc_block, self).__init__()

        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_channels, out_channels))
        if norm:
            block.add_module('batchnorm', nn.BatchNorm1d(out_channels))
        if activation_fn is not None:
            block.add_module('activation', activation_fn())

        self.block = block

    def forward(self, x):
        return self.block(x)