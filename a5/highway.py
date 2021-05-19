#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embedding_size=128, gate_bias=-2):
        super(Highway, self).__init__()
        self.projection_layer = nn.Linear(word_embedding_size, word_embedding_size)
        self.gate_layer = nn.Linear(word_embedding_size, word_embedding_size, bias=True)
        self.gate_layer.bias.data.fill_(gate_bias) # See if this fits with the initialization or not...
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward layer of the module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with size S x B x E shape, being S the 
            sentence length, B the batch size and E the 
            embedding of the word.

        Returns
        -------
        output_tensor : torch.Tensor
            Tensor, output of Highway module, with size S x B x E shape, 
            being S the sentence length, B the 
            batch size and E the embedding of the word.

        """
        x_proj = self.relu(self.projection_layer(x))
        x_gate = self.sigmoid(self.gate_layer(x))
        output_tensor = x_gate * x_proj + (1 - x_gate) * x
        return output_tensor

    ### END YOUR CODE

