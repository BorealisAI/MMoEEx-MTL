"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Task balacing approaches for multi-task learning

Two methods based on loss ratio called:
- DWA - Dynamic Weight Average
- LBTW - Loss Balanced Task Weighting
Written by Gabriel Oliveira in pytorch

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint, binomial
import numpy as np


class TaskBalanceMTL:
    # Class for task balancing methods
    def __init__(
        self, n_tasks, balance="None", balance_method="Sum", K=4, T=4, alpha_balance=0.5
    ):
        """[init Task Balance module]

        Args:
            n_tasks ([int]): [Number of tasks ]
            balance (str, optional): [balance algorithm [DWA,LBTW]]. Defaults to "None".
            balance_method (str, optional): [Now is some, made to be extended]. Defaults to "Sum".
            K (int, optional): [DWA task balance multiplier- all weight sum to K]. Defaults to 4.
            T (int, optional): [Division parameter T]. Defaults to 4.
            alpha_balance (float, optional): [hyper-parameter for LBTW]. Defaults to 0.5.
        """

        # Hyper parameters
        self.balance_method = balance_method
        self.K = n_tasks
        self.T = n_tasks
        self.alpha_balance = alpha_balance
        self.n_tasks = n_tasks
        self.task_ratios = np.zeros([self.n_tasks], dtype=np.float32)
        self.task_weights = np.zeros([self.n_tasks], dtype=np.float32)
        self.initial_losses = np.zeros([self.n_tasks], dtype=np.float32)
        self.weight_history = []
        self.history_last = []
        for i in range(self.n_tasks):
            self.weight_history.append([])
            self.history_last.append([])

        # Setting weight method
        self.balance_mode = balance
        if self.balance_mode == "DWA":
            print("...DWA Weight balance")
        if self.balance_mode == "LBTW":
            print("...LBTW Weight balance")

    def add_loss_history(self, task_losses):
        for i in range(0, self.n_tasks):
            self.weight_history[i].append(task_losses[i])

    def last_elements_history(self):
        for i in range(0, self.n_tasks):
            self.history_last[i] = self.weight_history[i][-2:]

    def compute_ratios(self, task_losses, epoch):

        for i in range(0, self.n_tasks):
            if epoch <= 1:
                self.task_ratios[:] = 1
            else:
                before = "-"
                if self.history_last[i][-2] > -0.01 and self.history_last[i][-2] < 0.01:
                    before = self.history_last[i][-2]
                    self.history_last[i][-2] = 0.01

                self.task_ratios[i] = (
                    self.history_last[i][-1] / self.history_last[i][-2]
                )

    def sum_losses_tasks(self):
        ratios_sum = 0.0
        for i in range(0, self.n_tasks):
            ratios_sum += np.exp(self.task_ratios[i] / self.T)
        return ratios_sum

    def DWA(self, task_losses, epoch):
        self.compute_ratios(task_losses, epoch)
        ratios_sum = self.sum_losses_tasks()

        for i in range(0, self.n_tasks):
            self.task_weights[i] = max(
                min((self.K * np.exp(self.task_ratios[i] / self.T)) / ratios_sum, 1.5),
                0.5,
            )

    def get_weights(self):
        return self.task_weights

    def get_initial_loss(self, losses, task):
        self.initial_losses[task] = losses

    def LBTW(self, batch_losses, task):
        self.task_weights[task] = max(
            min(pow(batch_losses / self.initial_losses[task], self.alpha_balance), 1.5),
            0.5,
        )
