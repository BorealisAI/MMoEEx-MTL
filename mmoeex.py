"""
Copyright (c) 2020-present, Royal Bank of Canada.
Copyright (c) 2020-present, Alvin Deng
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Code is based on the MMoE (Modeling task relationships in multi-task learning with multi-gate mixture-of-experts) implementation 
from https://github.com/drawbridge/keras-mmoe by Alvin Deng

Written by Raquel Aoki and Gabriel Oliveira
Date: December 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
import numpy as np
import sys

sys.path.append("/src")


class MMoETowers(nn.Module):
    def __init__(
        self,
        data,
        tasks_name,
        num_tasks,
        num_experts,
        num_units,
        num_features,
        modelname,
        task_info=None,
        task_number=None,
        runits=None,
        expert=None,
        expert_blocks=None,
        seqlen=None,
        n_layers=1,
        prob_exclusivity=0.5,
        type="exclusivity",
    ):
        super(MMoETowers, self).__init__()
        self.data = data
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.modelname = modelname
        self.runits = runits
        self.seqlen = seqlen
        self.tasks_name = tasks_name
        self.num_units = num_units
        self.type = type
        self.task_number = task_number
        self.task_info = task_info

        if modelname == "MMoE":
            self.MMoE = MMoE(
                data=data,
                units=num_units,
                num_experts=num_experts,
                num_tasks=num_tasks,
                num_features=num_features,
                use_expert_bias=True,
                use_gate_bias=True,
                runits=runits,
                expert=expert,
                expert_blocks=expert_blocks,
                seqlen=seqlen,
                n_layers=n_layers,
            )
        elif modelname == "MMoEEx" or modelname == "Md":
            self.MMoEEx = MMoEEx(
                data=data,
                units=num_units,
                num_experts=num_experts,
                num_tasks=num_tasks,
                num_features=num_features,
                use_expert_bias=True,
                use_gate_bias=True,
                runits=runits,
                expert=expert,
                expert_blocks=expert_blocks,
                seqlen=seqlen,
                n_layers=n_layers,
                prob_exclusivity=prob_exclusivity,
                type=type,
            )

        if self.task_info is not None:
            # MIMIC Dataset
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

        if self.seqlen is None:
            tower = self.num_experts
        else:
            tower = self.runits[0]

        if self.data == "pcba":
            self.towers_list = nn.ModuleList(
                [nn.Identity(tower, self.num_units) for i in range(self.num_tasks)]
            )
            self.output_list = nn.ModuleList(
                [nn.Linear(self.num_experts, out[i]) for i in range(self.num_tasks)]
            )
        else:
            self.towers_list = nn.ModuleList(
                [nn.Linear(tower, self.num_units) for i in range(self.num_tasks)]
            )
            self.output_list = nn.ModuleList(
                [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
            )

    def forward(self, input, params=None, diversity=False):
        input = input.float()

        if params is not None:
            for (_1, p), (_2, p_) in zip(params.items(), self.named_parameters()):
                p_.data = p.data

        if self.modelname == "MMoE" or self.modelname == "Mm":
            if diversity:
                x, div = self.MMoE(input, diversity=True)
            else:
                x = self.MMoE(input, diversity=False)  # T x N x E
        elif self.modelname == "MMoEEx" or self.modelname == "Md":
            if diversity:
                x, div = self.MMoEEx(input, diversity=True)
            else:
                x = self.MMoEEx(input, diversity=False)

        output = []
        for task in range(self.num_tasks):
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

        if diversity:
            return div
        else:
            return output

    def mimic_fix_task_time(self, data, task):
        if self.seqlen is not None:
            if self.tasks_name[task] == "ihm":
                data = data[:, 0:24]
        return data


class MMoEEx(nn.Module):
    def __init__(
        self,
        data,
        units,
        num_experts,
        num_tasks,
        num_features,
        use_expert_bias=True,
        use_gate_bias=True,
        runits=None,
        expert=None,
        expert_blocks=None,
        seqlen=None,
        n_layers=1,
        prob_exclusivity=0.5,
        type="exclusivity",
    ):
        super(MMoEEx, self).__init__()
        self.data = data
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.seqlen = seqlen
        self.expert = expert
        self.expert_blocks = expert_blocks
        self.n_layers = n_layers
        self.prob_exclusivity = prob_exclusivity
        self.type = type  # exclusivity or exclusion

        """Creating exclusivity or exclusion array"""
        exclusivity = np.repeat(self.num_tasks + 1, self.num_experts)
        to_add = int(self.num_experts * self.prob_exclusivity)
        for e in range(to_add):
            exclusivity[e] = randint(0, self.num_tasks)
        self.exclusivity = exclusivity
        if self.data == "pcba" and self.type == "exclusivity":
            exclusivity[0] = 125

        print("... Model - MMOEEx")
        print("... ", self.type, ":", exclusivity)

        self.runits = runits
        """Defining the experts: right now, all experts have the same architecture"""
        if self.expert is None:
            self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.rand(size=(self.num_features, self.runits[0])).float()
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.runits[0], 1).float()
                    for i in range(self.num_tasks * self.num_experts)
                ]
            )
        elif self.expert == "Expert_2layers":
            print("... Expert with 2 layers")
            self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.rand(size=(self.num_features, self.runits[0])).float()
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_midlayer = nn.ModuleList(
                [
                    nn.Linear(self.runits[0], self.runits[0]).float()
                    for i in range(self.num_experts)
                ]
            )

            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.runits[0], self.num_tasks).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_LSTM":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_LSTM(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Residual_LSTM":
            self.expert_kernels = nn.ModuleList(
                [
                    Residual_LSTM(
                        self.runits[0],
                        self.seqlen,
                        self.num_features,
                        self.n_layers,
                        self.expert_blocks,
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_RNN":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_RNN(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_GRU":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_GRU(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        if self.seqlen is None:
            gate_kernels = torch.rand(
                (self.num_tasks, self.num_features, self.num_experts)
            ).float()
        else:
            gate_kernels = torch.rand(
                (self.num_tasks, self.seqlen * self.num_features, self.num_experts)
            ).float()

        """ Initialize expert bias (number of units per expert * number of experts)# Bias parameter"""
        if use_expert_bias:
            if self.seqlen is None:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts), requires_grad=True
                )
            else:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts, self.seqlen), requires_grad=True
                )

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(
                torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True
            )

        """
        Setting the exclusivity
         t: which task has the expert (all other should be set to 0)
         e: which expert is exclusive
        """
        for e, t in enumerate(self.exclusivity):
            if t < self.num_tasks + 1:
                if self.type == "exclusivity":
                    for tasks in range(self.num_tasks):
                        if tasks != t:
                            gate_kernels[tasks][:, e] = 0.0
                else:
                    gate_kernels[t][:, e] = 0.0

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        # References:
        # https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py

        n = inputs.shape[0]
        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)
        if self.data == "census":
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
                if i == 0:
                    expert_outputs = self.expert_output[i](aux)
                else:
                    expert_outputs = torch.cat(
                        (expert_outputs, self.expert_output[i](aux)), dim=1
                    )
        elif self.data == "pcba":
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = self.dropout(aux)
                aux = F.relu(aux)
                aux = self.expert_midlayer[i](aux)
                torch.sigmoid(aux)
                aux = aux.reshape(1, aux.shape[0], aux.shape[1])
                if i == 0:
                    expert_outputs = aux
                else:
                    expert_outputs = torch.cat((expert_outputs, aux), dim=0)
        elif self.data == "mimic":
            # LSTM
            for i in range(self.num_experts):
                aux = self.expert_kernels[i](inputs)
                if i == 0:
                    expert_outputs = torch.reshape(
                        aux, (1, aux.shape[0], aux.shape[1], aux.shape[2])
                    )
                else:
                    expert_outputs = torch.cat(
                        (
                            expert_outputs,
                            torch.reshape(
                                aux, (1, aux.shape[0], aux.shape[1], aux.shape[2])
                            ),
                        ),
                        dim=0,
                    )
            inputs = inputs.reshape(
                (inputs.shape[0], inputs.shape[1] * inputs.shape[2])
            )
        else:
            print("...warning: new dataset might need to be updated here")
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
                if i == 0:
                    expert_outputs = self.expert_output[i](aux)
                else:
                    expert_outputs = torch.cat(
                        (expert_outputs, self.expert_output[i](aux)), dim=1
                    )

        if self.use_expert_bias:
            if self.data == "census":
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
                expert_outputs = F.relu(expert_outputs)
            elif self.data == "pcba":
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
            elif self.data == "mimic":
                for expert in range(self.num_experts):
                    s = self.expert_bias[expert]
                    expert_outputs[expert] = expert_outputs[expert].add(s[:, None])
            else:
                print("...warning: new dataset might need to be updated here")
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
                expert_outputs = F.relu(expert_outputs)

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(
                    1, n, self.num_experts
                )

            else:
                gate_outputs = torch.cat(
                    (
                        gate_outputs,
                        torch.mm(inputs, self.gate_kernels[index]).reshape(
                            1, n, self.num_experts
                        ),
                    ),
                    dim=0,
                )

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)

        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""
        if self.data == "census":
            for task in range(self.num_tasks):
                gate = gate_outputs[task]
                final_outputs_t = torch.mul(gate, expert_outputs).reshape(
                    1, gate.shape[0], gate.shape[1]
                )
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                if task == 0:
                    final_outputs = final_outputs_t
                else:
                    final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

        elif self.data == "pcba":
            for ex in range(self.num_experts):
                if ex == 0:
                    a = self.expert_output[ex](expert_outputs[ex]).shape
                    expert_outputs_ = self.expert_output[ex](
                        expert_outputs[ex]
                    ).reshape(1, a[0], a[1])
                else:
                    expert_outputs_ = torch.cat(
                        (
                            expert_outputs_,
                            self.expert_output[ex](expert_outputs[ex]).reshape(
                                1, a[0], a[1]
                            ),
                        ),
                        dim=0,
                    )
            expert_outputs_ = expert_outputs_.transpose(0, 2)
            final_outputs = torch.mul(gate_outputs, expert_outputs_)
        elif self.data == "mimic":
            for task in range(self.num_tasks):
                # Gate x Expert
                # Select the gate for the current task
                gate = gate_outputs[task]
                # time series data
                for expert in range(self.num_experts):
                    gate0 = gate[:, expert]
                    final_outputs0 = expert_outputs[expert] * gate0[:, None, None]
                    if expert == 0:
                        final_outputs_t = final_outputs0
                    else:
                        final_outputs_t = final_outputs_t.add(final_outputs0)

                # n x seq x units
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                s0, s1, s2 = final_outputs_t.shape
                if task == 0:
                    final_outputs = final_outputs_t.reshape(1, s0, s1, s2)
                else:
                    final_outputs = torch.cat(
                        (final_outputs, final_outputs_t.reshape(1, s0, s1, s2)), dim=0
                    )
        else:
            print("...warning: new dataset might need to be updated here")
            for task in range(self.num_tasks):
                gate = gate_outputs[task]
                final_outputs_t = torch.mul(gate, expert_outputs).reshape(
                    1, gate.shape[0], gate.shape[1]
                )
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                if task == 0:
                    final_outputs = final_outputs_t
                else:
                    final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

        # T x n x E
        if diversity:
            if self.data == "pcba":
                expert_outputs = expert_outputs_.transpose(0, 2)
            return final_outputs, expert_outputs
        else:
            return final_outputs


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts demo with census income data.
    Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts."
    Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.

    The code is based on the TensorFlow implementation:
    Reference: https://github.com/drawbridge/keras-mmoe
    """

    def __init__(
        self,
        data,
        units,
        num_experts,
        num_tasks,
        num_features,
        use_expert_bias=True,
        use_gate_bias=True,
        runits=None,
        expert=None,
        expert_blocks=None,
        seqlen=None,
        n_layers=1,
    ):

        super(MMoE, self).__init__()
        self.data = data
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.seqlen = seqlen
        self.expert = expert
        self.n_layers = n_layers

        print("... Model - MMOE")

        self.runits = runits
        """Defining the experts: right now, all experts have the same architecture"""
        if self.expert is None:
            print("... Expert - None")
            self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.rand(size=(self.num_features, self.runits[0])).float()
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [nn.Linear(self.runits[0], 1).float() for i in range(self.num_experts)]
            )

        elif self.expert == "Expert_2layers":
            print("... Expert with 2 layers")
            self.expert_kernels = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.rand(size=(self.num_features, self.runits[0])).float()
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_midlayer = nn.ModuleList(
                [
                    nn.Linear(self.runits[0], self.runits[0]).float()
                    for i in range(self.num_experts)
                ]
            )

            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.runits[0], self.num_tasks).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_LSTM":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_LSTM(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Residual_LSTM":
            self.expert_kernels = nn.ModuleList(
                [
                    Residual_LSTM(
                        self.runits[0],
                        self.seqlen,
                        self.num_features,
                        self.n_layers,
                        self.expert_blocks,
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_RNN":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_RNN(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )
        elif self.expert == "Expert_GRU":
            self.expert_kernels = nn.ModuleList(
                [
                    Expert_GRU(
                        self.runits[0], self.seqlen, self.num_features, self.n_layers
                    )
                    for i in range(self.num_experts)
                ]
            )
            self.expert_output = nn.ModuleList(
                [
                    nn.Linear(self.seqlen, self.seqlen).float()
                    for i in range(self.num_experts)
                ]
            )

        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        if self.seqlen is None:
            gate_kernels = torch.rand(
                (self.num_tasks, self.num_features, self.num_experts)
            ).float()
        else:
            gate_kernels = torch.rand(
                (self.num_tasks, self.seqlen * self.num_features, self.num_experts)
            ).float()

        """Initialize expert bias (number of units per expert * number of experts)# Bias parameter"""
        if use_expert_bias:
            if self.seqlen is None:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts), requires_grad=True
                )
            else:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts, self.seqlen), requires_grad=True
                )

        """ Initialize gate bias (number of experts * number of tasks)"""
        if use_gate_bias:
            self.gate_bias = nn.Parameter(
                torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True
            )

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, diversity=False):
        # https://github.com/drawbridge/keras-mmoe/blob/master/mmoe.py
        # https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py
        # https://towardsdatascience.com/different-types-of-regularization-on-neuronal-network-with-pytorch-a9d6faf4793e

        n = inputs.shape[0]
        """ Calculating the experts """
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper (E x n x U)
        if self.data == "census":
            # census data, traditional approach from MMOE
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
                if i == 0:
                    expert_outputs = self.expert_output[i](aux)
                else:
                    expert_outputs = torch.cat(
                        (expert_outputs, self.expert_output[i](aux)), dim=1
                    )
        elif self.data == "pcba":
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = self.dropout(aux)
                aux = F.relu(aux)
                aux = self.expert_midlayer[i](aux)
                torch.sigmoid(aux)
                aux = aux.reshape(1, aux.shape[0], aux.shape[1])

                if i == 0:
                    expert_outputs = aux
                else:
                    expert_outputs = torch.cat((expert_outputs, aux), dim=0)
        elif self.data == "mimic":
            # LSTM
            for i in range(self.num_experts):
                aux = self.expert_kernels[i](inputs)
                if i == 0:
                    expert_outputs = torch.reshape(
                        aux, (1, aux.shape[0], aux.shape[1], aux.shape[2])
                    )
                else:
                    expert_outputs = torch.cat(
                        (
                            expert_outputs,
                            torch.reshape(
                                aux, (1, aux.shape[0], aux.shape[1], aux.shape[2])
                            ),
                        ),
                        dim=0,
                    )
            inputs = inputs.reshape(
                (inputs.shape[0], inputs.shape[1] * inputs.shape[2])
            )
        else:
            print("...warning: new dataset might need to be updated here")
            for i in range(self.num_experts):
                aux = torch.mm(inputs, self.expert_kernels[i])
                aux = torch.reshape(aux, (n, self.expert_kernels[i].shape[1]))
                if i == 0:
                    expert_outputs = self.expert_output[i](aux)
                else:
                    expert_outputs = torch.cat(
                        (expert_outputs, self.expert_output[i](aux)), dim=1
                    )

        if self.use_expert_bias:
            if self.data == "census":
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
                expert_outputs = F.relu(expert_outputs)
            elif self.data == "pcba":
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
            elif self.data == "mimic":
                for expert in range(self.num_experts):
                    s = self.expert_bias[expert]
                    expert_outputs[expert] = expert_outputs[expert].add(s[:, None])
            else:
                print("...warning: new dataset might need to be updated here")
                for i in range(self.num_experts):
                    expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
                expert_outputs = F.relu(expert_outputs)

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(inputs, self.gate_kernels[index]).reshape(
                    1, n, self.num_experts
                )

            else:
                gate_outputs = torch.cat(
                    (
                        gate_outputs,
                        torch.mm(inputs, self.gate_kernels[index]).reshape(
                            1, n, self.num_experts
                        ),
                    ),
                    dim=0,
                )

        if self.use_gate_bias:
            gate_outputs = gate_outputs.add(self.gate_bias)

        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""
        if self.data == "census":
            for task in range(self.num_tasks):
                gate = gate_outputs[task]
                final_outputs_t = torch.mul(gate, expert_outputs).reshape(
                    1, gate.shape[0], gate.shape[1]
                )
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                if task == 0:
                    final_outputs = final_outputs_t
                else:
                    final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)
        elif self.data == "pcba":
            for ex in range(self.num_experts):
                if ex == 0:
                    a = self.expert_output[ex](expert_outputs[ex]).shape
                    expert_outputs_ = self.expert_output[ex](
                        expert_outputs[ex]
                    ).reshape(1, a[0], a[1])
                else:
                    expert_outputs_ = torch.cat(
                        (
                            expert_outputs_,
                            self.expert_output[ex](expert_outputs[ex]).reshape(
                                1, a[0], a[1]
                            ),
                        ),
                        dim=0,
                    )
            expert_outputs_ = expert_outputs_.transpose(0, 2)
            final_outputs = torch.mul(gate_outputs, expert_outputs_)
        elif self.data == "mimic":
            # time series
            for task in range(self.num_tasks):
                # Gate x Expert
                # Select the gate for the current task
                gate = gate_outputs[task]
                for expert in range(self.num_experts):
                    # operation to fix the dim
                    gate0 = gate[:, expert]
                    # current expert x gate row-wise (n x seq x u)*(n x none) = (n x seq)
                    final_outputs0 = expert_outputs[expert] * gate0[:, None, None]
                    if expert == 0:
                        final_outputs_t = final_outputs0
                    else:
                        final_outputs_t = final_outputs_t.add(final_outputs0)

                # n x seq x units
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                s0, s1, s2 = final_outputs_t.shape
                if task == 0:
                    final_outputs = final_outputs_t.reshape(1, s0, s1, s2)
                else:
                    final_outputs = torch.cat(
                        (final_outputs, final_outputs_t.reshape(1, s0, s1, s2)), dim=0
                    )
        else:
            print("...warning: new dataset might need to be updated here")
            for task in range(self.num_tasks):
                gate = gate_outputs[task]
                final_outputs_t = torch.mul(gate, expert_outputs).reshape(
                    1, gate.shape[0], gate.shape[1]
                )
                final_outputs_t = final_outputs_t.add(self.task_bias[task])
                if task == 0:
                    final_outputs = final_outputs_t
                else:
                    final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

        # T x n x E
        if diversity:
            if self.data == "pcba":
                expert_outputs = expert_outputs_.transpose(0, 2)
            return final_outputs, expert_outputs
        else:
            return final_outputs


class Expert_LSTM(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def __init__(self, units, seqlen, num_features, n_layers):
        super(Expert_LSTM, self).__init__()
        self.units = units
        self.seqlen = seqlen
        self.num_features = num_features
        self.n_layers = n_layers
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.units,
            num_layers=self.n_layers,
            dropout=0.3,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=self.num_features,
            hidden_size=self.units,
            num_layers=self.n_layers,
            dropout=0.3,
            batch_first=True,
        )
        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(self.units, self.units)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.tanh(lstm_out)
        pred = self.hidden(lstm_out)
        return pred


class Expert_RNN(nn.Module):
    def __init__(self, units, seqlen, num_features, n_layers):
        super(Expert_RNN, self).__init__()
        # Hidden nodes parameter (float/int)
        self.units = units
        self.seqlen = seqlen
        self.num_features = num_features
        self.n_layers = n_layers
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # default dropout 0.3 - testing 0.5
        self.rnn = nn.RNN(
            input_size=self.num_features,
            hidden_size=self.units,
            num_layers=self.n_layers,
            dropout=0.5,
            batch_first=True,
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(self.units, self.units)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rnn_out, _ = self.rnn(inputs)
        rnn_out = self.tanh(rnn_out)
        pred = self.hidden(rnn_out)
        return pred


class Expert_GRU(nn.Module):
    def __init__(self, units, seqlen, num_features, n_layers):
        super(Expert_GRU, self).__init__()
        # Hidden nodes parameter (float/int)
        self.units = units
        self.seqlen = seqlen
        self.num_features = num_features
        self.n_layers = n_layers
        self.device = torch.device("cuda:0")
        """ The LSTM takes word embeddings as inputs, and outputs hidden states
        with dimensionality hidden_dim. default dropout 0.3 - testing 0.5 """
        self.GRU = nn.GRU(
            input_size=self.num_features,
            hidden_size=self.units,
            num_layers=self.n_layers,
            dropout=0.5,
            batch_first=True,
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden = nn.Linear(self.units, self.units)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        GRU_out, _ = self.GRU(inputs)
        GRU_out = self.tanh(GRU_out)
        pred = self.hidden(GRU_out)
        return pred
