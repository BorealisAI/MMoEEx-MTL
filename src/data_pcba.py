"""
Copyright (c) 2020-present, Royal Bank of Canada.
All rights reserved.
 This source code is licensed under the license found in the
 LICENSE file in the root directory of this source tree.

Routine to pre-process PCBA tasks

It requires a new virtualenv due to compatilities issues:

conda create -c rdkit -n my-rdkit-env rdkit
source activate env-pcba
pip install tensorflow-gpu==1.14
conda install -y -c conda-forge rdkit deepchem==2.3.0
conda install -c conda-forge tensorboard


"""
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor

sys.path.append("/src")
from data_preprocessing import *

print("Warning: it requires a different enviroment.\n")
print(
    "Instructions\n: conda create -c rdkit -n my-rdkit-env rdkit\nsource activate env-pcba\npip install tensorflow-gpu==1.14\nconda install -y -c conda-forge rdkit deepchem==2.3.0\nconda install -c conda-forge tensorboard\n"
)

run_dataprep()

print("DONE!")
