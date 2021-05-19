#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        output =  self.decoderCharEmb(input) # (L, B, C_E)
        output, (h_t, c_t) = self.charDecoder(output, dec_hidden) # (L, B, H)
        scores = self.char_output_projection(output) # (L, B, H) * (H, V) = (L, B, V)
        return scores, (h_t, c_t)
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None): # TODO: Review this!!!
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        scores, (_, _) = self.forward(char_sequence[:-1, :], dec_hidden=dec_hidden) # (length, batch_size, self.vocab_size)
        target = char_sequence[1:, :] # (L-1) x B
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad, reduction='sum')(scores.permute((1, 2, 0)), target.permute((1, 0)))
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21): # TODO: Review
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        word_length = 0
        batch_size = initialStates[0].shape[1]
        decodedWords = ["" for _ in range(batch_size)]
        start_token = torch.tensor([self.target_vocab.start_of_word], dtype=torch.int64, device=device).unsqueeze(-1)
        input = torch.repeat_interleave(start_token, batch_size, dim=-1)
        while word_length < max_length:           
            logits, (h_t, c_t) = self.forward(input, initialStates) # Input: 1 x B ; Output: 1 x B
            initialStates = (h_t, c_t)  # Feedback
            scores = torch.softmax(logits, dim=-1)
            top_pos = torch.argmax(scores, dim=-1)
            input = top_pos # Feedback
            for idx, top_pos_item in enumerate(torch.flatten(top_pos).cpu().numpy()):
                decodedWords[idx] += self.target_vocab.id2char[top_pos_item]
            word_length += 1

        filtereddecodedWords = list()
        for word in decodedWords:
            filtered_word = ""
            for c in word:
                if c == self.target_vocab.id2char[self.target_vocab.end_of_word]:
                    break
                filtered_word += c
            filtereddecodedWords.append(filtered_word)
        return filtereddecodedWords
        ### END YOUR CODE

