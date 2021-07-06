"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Standard Single optimization step
Shared Bottom
Written by Gabriel Oliveira and Raquel Aoki in pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
from residual_lstm import Residual_LSTM
from mmoe import Expert_LSTM
import numpy as np
import sys


class standarOptimization(nn.Module):
    def __init__(
        self,
        data,
        num_units,
        tasks_name,
        num_experts,
        num_tasks,
        num_features,
        hidden_size,
        learning_type=None,
        seqlen=None,
        n_layers=1,
        task_number=None,
        expert=None,
        task_info=None,
        runits=None,
    ):
        super(standarOptimization, self).__init__()
        # Hidden nodes parameter (float/int)
        self.data = data
        self.num_units = num_units
        self.hidden_size = hidden_size
        self.num_experts = 1
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.task_number = task_number
        self.expert = expert
        self.task_info = task_info
        self.runits = runits
        self.tasks_name = tasks_name
        self.seqlen = seqlen
        self.n_layers = n_layers
        self.learning_type = learning_type

        print("...Optimization - Standard optimization / shared bottom")

        self.sharedBottom = sharedBottom(
            data=self.data,
            num_units=self.num_units,
            num_experts=self.num_experts,
            num_tasks=self.num_tasks,
            num_features=self.num_features,
            hidden_size=self.hidden_size,
            learning_type=self.learning_type,
            seqlen=self.seqlen,
            task_number=self.task_number,
            expert=self.expert,
            runits=self.runits,
        )

        if self.data == "mimic":
            out = [task_info[t] for t in tasks_name]
            inp = []
            for t in tasks_name:
                if t == "pheno":
                    inp.append(self.seqlen * self.num_units)
                elif t == "ihm":
                    inp.append(24 * self.num_units)
                else:
                    inp.append(self.num_units)
        else:
            out = [1 for t in range(num_tasks)]
            inp = [self.num_units for t in range(num_tasks)]

        tower = self.runits[0]
        if self.data == "pcba":
            self.num_units = 1

        self.towers_list = nn.ModuleList(
            [nn.Linear(tower, self.num_units) for i in range(self.num_tasks)]
        )
        if self.data == "pcba":
            self.output_list = nn.ModuleList(
                [nn.Identity(inp[i], out[i]) for i in range(self.num_tasks)]
            )
        else:
            self.output_list = nn.ModuleList(
                [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
            )
        self.dropout = nn.Dropout(0.25)

    def forward(self, input, params=None, ts=None):
        input = input.float()
        if params is not None:
            for (name, p), (name_, p_) in zip(params.items(), self.named_parameters()):
                p_.data = p.data

        x = self.sharedBottom(input)  # T x N x E
        output = []
        for task in range(self.num_tasks):
            if self.seqlen is None:
                aux = x
            else:
                aux = self.mimic_fix_task_time(x[task], task)
            aux = self.towers_list[task](aux)  # n x seq x new_units
            if self.tasks_name[task] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
            elif self.tasks_name[task] == "pheno":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
            aux = self.output_list[task](aux)
            output.append(aux)

        return output

    def mimic_fix_task_time(self, data, task):
        if self.seqlen is not None:
            if self.tasks_name[task] == "ihm":
                data = data[:, 0:24]
        return data


class sharedBottom(nn.Module):
    def __init__(
        self,
        data,
        num_units,
        num_experts,
        num_tasks,
        num_features,
        learning_type,
        hidden_size,
        seqlen=None,
        n_layers=1,
        task_number=None,
        expert=None,
        runits=None,
    ):
        """[summary]

        Args:
            units ([type]): [description]
            num_experts ([type]): [description]
            num_tasks ([type]): [description]
            num_features ([type]): [description]
            learning_type ([type]): [description]
            hidden_size ([type]): [description]
            task_number ([type]): [description]
            seqlen ([type], optional): [description]. Defaults to None.
            n_layers (int, optional): [description]. Defaults to 1.
        """

        super(sharedBottom, self).__init__()
        # Hidden nodes parameter (float/int)
        self.num_units = num_units
        self.data = data
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.task_number = task_number
        self.expert = expert
        self.runits = runits
        self.seqlen = seqlen
        self.n_layers = n_layers
        self.learning_type = learning_type
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
        if self.learning_type == "LSTM":
            if self.hidden_size is None:
                self.hidden_size = np.repeat(256, self.num_experts)
            self.expert_kernels = nn.LSTM(
                input_size=self.num_features,
                hidden_size=hidden_size,
                num_layers=self.n_layers,
                dropout=0.3,
                batch_first=True,
            )
            self.expert_output = nn.Linear(self.seqlen, self.seqlen).float()
        elif self.learning_type is None:
            self.expert_kernels = nn.Parameter(
                torch.rand(size=(self.num_features, self.runits[0])).float()
            )
            self.expert_output = nn.ModuleList([nn.Identity(self.runits[0]).float()])
        elif self.learning_type is "Expert_2layers":
            self.expert_kernels = nn.Parameter(
                torch.rand(size=(self.num_features, self.runits[0])).float()
            )
            self.expert_output = nn.ModuleList(
                [nn.Linear(self.runits[0], self.runits[0]).float()]
            )

    def forward(self, inputs):

        n = inputs.shape[0]
        if self.learning_type is None:
            aux = torch.mm(inputs, self.expert_kernels)
            aux = torch.reshape(aux, (n, self.expert_kernels.shape[1]))
            expert_outputs = torch.reshape(aux, (aux.shape[0], aux.shape[1]))
            expert_outputs = F.relu(expert_outputs)
            final_outputs = self.expert_output[0](expert_outputs)
        elif self.learning_type == "Expert_2layers":
            aux = torch.mm(inputs, self.expert_kernels)
            aux = torch.reshape(aux, (n, self.expert_kernels.shape[1]))
            expert_outputs = torch.reshape(aux, (aux.shape[0], aux.shape[1]))
            expert_outputs = self.dropout(expert_outputs)
            expert_outputs = F.relu(expert_outputs)
            final_outputs = self.expert_output[0](expert_outputs)
            final_outputs = self.sigmoid(final_outputs)
        else:
            # LSTM
            inputs = inputs.float()
            aux, _ = self.expert_kernels(inputs)
            aux = aux.reshape(1, aux.shape[0], aux.shape[1], aux.shape[2])
            expert_outputs = aux
            inputs = inputs.reshape(
                (inputs.shape[0], inputs.shape[1] * inputs.shape[2])
            )

            for task in range(self.num_tasks):
                if task == 0:
                    final_outputs = expert_outputs
                else:
                    final_outputs = torch.cat((final_outputs, expert_outputs), dim=0)

        return final_outputs
