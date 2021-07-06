"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Residual LSTM class 
Method to add residual connections to LSTM class
Written by Gabriel Oliveira in pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
import numpy as np


class Residual_LSTM(nn.Module):
    def __init__(self, units, seqlen, num_features, n_layers, n_blocks):
        """[summary]

        Args:
            units ([int]): [description]
            seqlen ([int]): [description]
            num_features ([int]): [description]
            n_layers ([int]): [description]
            n_blocks ([int]): [description]
        """
        super(Residual_LSTM, self).__init__()
        self.n_blocks = n_blocks
        self.units = units
        self.seqlen = seqlen
        self.num_features = num_features
        self.n_layers = n_layers
        self.lstm_blocks = []
        self.dropout_layer = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        # LSTM layers
        self.__build_residual_network()
        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(self.units, self.units)
        self.tanh = nn.Tanh()

    def __build_residual_network(self):
        for i in range(0, self.n_blocks):
            if i == 0:
                self.lstm_blocks.append(
                    nn.LSTM(
                        input_size=self.num_features,
                        hidden_size=self.units,
                        num_layers=self.n_layers,
                        dropout=0.3,
                        batch_first=True,
                    )
                )
            else:
                self.lstm_blocks.append(
                    nn.LSTM(
                        input_size=self.units,
                        hidden_size=self.units,
                        num_layers=self.n_layers,
                        dropout=0.3,
                        batch_first=True,
                    )
                )
        self.lstm_blocks = torch.nn.ModuleList(self.lstm_blocks)

    def forward(self, inputs):

        for i in range(0, self.n_blocks):
            if i == 0:
                current_lstm = self.lstm_blocks[i]
                y, _ = current_lstm(inputs)
                skip_connection = y
            elif i < (self.n_blocks - 1):
                current_lstm = self.lstm_blocks[i]
                y, _ = current_lstm(y)
                # skip connection
                skip_connection_dropped = self.dropout_layer(skip_connection)
                fuse = self.relu(y.add(skip_connection_dropped))
                skip_connection = fuse
                y = fuse
            else:
                current_lstm = self.lstm_blocks[i]
                y, _ = current_lstm(y)
                # skip connection
                skip_connection_dropped = self.dropout_layer(skip_connection)
                fuse = self.relu(y.add(skip_connection_dropped))
                y = fuse

        lstm_out = self.tanh(y)
        pred = self.hidden(lstm_out)
        return pred
