import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import torch
import numpy as np
# Importing Hopfield-specific modules.
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
from hflayers.auxiliary.data import BitPatternSet
from sparse_hflayers import SparseHopfield, SparseHopfieldPooling, SparseHopfieldLayer

# Import auxiliary modules.
from distutils.version import LooseVersion
from typing import List, Tuple
from prettytable import PrettyTable

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Flatten, Linear, Module, Sequential, MultiheadAttention
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from bit_numerical import *
import torch.nn as nn

def training(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            data_loader_valid: DataLoader,
            num_epochs: int = 1,
            note: str = '',
            lr_decay = 0.5
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    losses, accuracies = {f'train': [], r'eval': []}, {r'train': [], r'eval': []}
    best_val_acc = -1
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_valid)
        if performance[1] >= best_val_acc:
            best_val_acc = performance[1]
            p = eval_iter(network, data_loader_eval)
            best_acc = p[1]

        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
    
        for g in optimiser.param_groups:
            g['lr'] *= lr_decay

    return losses, accuracies

def single_run(config, num_bag=10, num_signals=1, mode='dense', rs=1111):

    set_seed(rs)
    train_loader, eval_loader, valid_loader, bit_pattern_set = get_data(num_instances=num_bag, batch_size=config["batch_size"], num_signals=int(num_signals))
    bit_samples_unique = [_[r'data'] for _ in train_loader]
    bit_samples_unique = torch.cat(bit_samples_unique).view(-1, bit_samples_unique[0].shape[2]).unique(dim=0)

    if mode == 'dense':
        model, opt = build_layer('standard', bit_pattern_set, bit_samples_unique, config)
    elif mode == 'sparse':
        model, opt = build_layer('sparse', bit_pattern_set, bit_samples_unique, config)
    elif mode == 'attn':
        model, opt = build_attn_layer(bit_pattern_set, bit_samples_unique, config)

    losses, accs = training(network=model, optimiser=opt, data_loader_train=train_loader, data_loader_eval=eval_loader, data_loader_valid=valid_loader, num_epochs=10, lr_decay=config["lr_decay"])

    return losses, accs

def plot_performance(loss: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_file: str,
                     train=False
                    ) -> None:
    """
    Plot and save loss and accuracy.
    
    :param loss: loss to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    sns.set(font_scale=2)

    # dense_loss = loss[col for col in loss.columns if 'dense' in col]
    # sparse_loss = loss[col for col in loss.columns if 'sparse' in col]

    # dense_acc = loss[col for col in accuracy.columns if 'dense' in col]
    # sparse_acc = loss[col for col in accuracy.columns if 'sparse' in col]

    dash = [0, 1]*5
    # dash = True

    if train:
        loss_plot = sns.lineplot(data=loss, ax=ax[0], palette=['b', 'b', 'r', 'r', 'orange', 'orange', 'green', 'green', 'purple', 'purple'], dashes=dash)
        loss_plot.set(xlabel=r'Epoch', ylabel=r'Training Loss')
        
        accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1], palette=['b', 'b', 'r', 'r', 'orange', 'orange', 'green', 'green', 'purple', 'purple'], dashes=dash)
        accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Training Accuracy')
    else:
        loss_plot = sns.lineplot(data=loss, ax=ax[0], palette=['b', 'b', 'r', 'r', 'orange', 'orange', 'green', 'green', 'purple', 'purple'], dashes=dash)
        loss_plot.set(xlabel=r'Epoch', ylabel=r'Validation Loss')
        
        accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1], palette=['b', 'b', 'r', 'r', 'orange', 'orange', 'green', 'green', 'purple', 'purple'], dashes=dash)
        accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Validation Accuracy')

    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    fig.savefig(log_file)
    # plt.show(fig)

def num_bag_exp():

    modes = ['dense', 'sparse', 'attn']

    config = {
        "lr": 1e-3,
        "lr_decay": 0.98,
        "batch_size": 64,
        "dropout": 0.0
    }

    num_bag = [5, 10]# , 20, 30, 40] # , 20, 30, 50]

    acc_results = {}
    loss_results = {}

    test_acc_results = {}
    test_loss_results = {}

    for n in num_bag:

        acc_results[f"dense M = {n}"] = np.zeros(10)
        loss_results[f"dense M = {n}"] = np.zeros(10)
        acc_results[f"sparse M = {n}"] = np.zeros(10)
        loss_results[f"sparse M = {n}"] = np.zeros(10)

        test_acc_results[f"dense M = {n}"] = np.zeros(10)
        test_loss_results[f"dense M = {n}"] = np.zeros(10)
        test_acc_results[f"sparse M = {n}"] = np.zeros(10)
        test_loss_results[f"sparse M = {n}"] = np.zeros(10)

        for i in range(1, 6):

            dense_loss, dense_acc = single_run(config=config, num_bag=n, mode='dense', rs=int(i*1111))
            sparse_loss, sparse_acc = single_run(config=config, num_bag=n, mode='sparse', rs=int(i*1111))

            loss_results[f"dense M = {n}"] += dense_loss[r'train']
            loss_results[f"sparse M = {n}"] += sparse_loss[r'train']

            acc_results[f"dense M = {n}"] += dense_acc[r'train']
            acc_results[f"sparse M = {n}"] += sparse_acc[r'train']

            test_loss_results[f"dense M = {n}"] += dense_loss[r'eval']
            test_loss_results[f"sparse M = {n}"] += sparse_loss[r'eval']

            test_acc_results[f"dense M = {n}"] += dense_acc[r'eval']
            test_acc_results[f"sparse M = {n}"] += sparse_acc[r'eval']

        loss_results[f"dense M = {n}"] /= 5
        loss_results[f"sparse M = {n}"] /= 5

        acc_results[f"dense M = {n}"] /= 5
        acc_results[f"sparse M = {n}"] /= 5

        test_loss_results[f"dense M = {n}"] /= 5
        test_loss_results[f"sparse M = {n}"] /= 5

        test_acc_results[f"dense M = {n}"] /= 5
        test_acc_results[f"sparse M = {n}"] /= 5
        
    train_df = pd.DataFrame(loss_results)
    test_df = pd.DataFrame(test_loss_results)

    train_df_acc = pd.DataFrame(acc_results)
    test_df_acc = pd.DataFrame(test_acc_results)

    plot_performance(train_df, train_df_acc, './training.pdf', train=True)
    plot_performance(test_df, test_df_acc, './test.pdf', train=False)


if __name__ == '__main__':

    num_bag_exp()