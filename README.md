# MMoEEx-MTL

This repository provides the codes for training and testing the Multi-Gated Mixture of Expert with Exclusivity (MMoEEx)
used for multi-task learning as described in the paper:

[Raquel Aoki, Frederick Tung, Gabriel L. Oliveira, Heterogeneous multi-task learning with expert diversity. BIOKDD 2021](http://arxiv.org/abs/2106.10595)

## How to Run

1. To run the code on terminal:

   Create a virtual environment using the these [requirements](requirements.txt), and run the code:
   ```
   python main.py config/config_file.yaml
   ```	

2. Config Files most import parameters:

   a. `tasks`: define the tasks the model will use and depends on the `data` being used.

   b. `models`: MMoE (no task exclusivity/exclusion, no MAML), Md (task exclusivity/exclusion, no MAML), and MMoEEx (
   task exclusivity/exclusion + MAML)

   c. `save_tensor`: boolean.

   d. `seqlen`, `prop` and `lambda`: Required only if `data` is `mimic`. `seqlen` is the maximum size of the
   sequences; `prop` is the proportion of the dataset being used; `lambda` is the loss weight.

We also added [four examples of config](config) files in our repo.

## Files Structure:

* `main.py`: call the data loaders, responsible for the training phase (loss functions, optimization);
* `data_preprocessing.py`: reads the data, performs all the preprocessing, creates and outputs the data loaders;
* `mmoeex.py`: it has the `torch.nn.Module`s for the models, implementation of MMoE, and MMoEEX;
* `utils.py`: support functions for main.py (organize outputs, calculate AUC values, gradients, etc).

## Datasets

Short description of the datasets

### Census

3 tasks:

- Income (+50000)
- marital status (is married)
- education (at least undergrad).

Training data (199,523), validation shape (49,881), test shape (49,881), and 482 features.

This dataset is public available, so our code download the datasets into the `mtl_datasets/census_dataset` folder. If
you already have the data downloaded in this folder, the code will only load the data.

### MIMIC3

4 tasks:

- IHM (in-hospital Mortality)
- decomp (decompensation, time-series)
- LOS (length-of-stay, time-series)
- pheno (phenotyping).

Training (28233), validation (6152), test (6056), **time-series** data, 76 features

This dataset is not public available, so you need to submit a request to work with this data
at [https://mimic.mit.edu/iii/gettingstarted/](https://mimic.mit.edu/iii/gettingstarted/). We followed the
pre-processing steps available
here: [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) and saved the
results in a folder named `mimic_dataset`.

### PCBA

128 tasks: 439863 molecules, with 1024 features. Each task is a biological target.

This dataset is public available. We provide this dataset in our repo because its pre-processing takes more time than
the Census dataset. This [Jupyter Notebook](resources/pcba_preprocessing.ipynb) shows from where to download and perform
the pre-processing.

## Citation

If you use MMoEEx-MTL, please consider citing:

```latex
@inproceedings{AokiBIOKDD21,
  author    = {Raquel Aoki and Frederick Tung and Gabriel L. Oliveira},
  title     = {{Heterogeneous Multi-task Learning with Expert Diversity}},
  booktitle = {BIOKDD},
  year      = {2021},
}
```
