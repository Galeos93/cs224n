#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, word_embedding_size, char_embedding_size):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv1d(char_embedding_size, word_embedding_size, 5, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        """Forward function of the module.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape S, B, E, W. Being S the sentence length, 
            B the batch size, W the max word length and 
            E the embedding size. W is the last dimension
            because PyTorch does convolution over last dimension.

        Return
        ------
        output : torch.Tensor
            Tensor with shape S, B, E. Being S the sentence length, 
            B the batch size and E the embedding size of the word.
        
        """
        sentence_length, batch_size, _, _ = x.shape
        output = torch.flatten(x, start_dim=0, end_dim=1)
        output = self.relu(self.conv_1(output)) # SxB, E, W
        output = self.max_pool(output)
        output = output.view(sentence_length, batch_size, output.size(1))    
        return output

    ### END YOUR CODE
