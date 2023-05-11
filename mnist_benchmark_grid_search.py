import os
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from hflayers import *
from sparse_hflayers import *
from datasets.loader import MnistBags

# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# from ray import tune
# from ray.air import session, RunConfig
# from ray.tune.schedulers import ASHAScheduler
import torch.nn as nn

from distutils.version import LooseVersion
from typing import Optional, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch


def get_args():

    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist'])
    parser.add_argument('--mode', default='standard', type=str, choices=['standard', 'sparse'])
    parser.add_argument('--rs', help='random state', default=1111, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

    parser.add_argument('--cpus_per_trial', default=4, type=int)
    parser.add_argument('--gpus_per_trial', default=0.0, type=float)
    parser.add_argument('--gpus_id', default="0", type=str)
    args = parser.parse_args()
    return args

device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

class HfPooling(nn.Module):    
    def __init__(self, mode='standard'):
        """
        Initialize a new instance of a Hopfield-based pooling network.
        
        Note: all hyperparameters of the network are fixed for demonstration purposes.
        Morevover, most of the notation of the original implementation is kept in order
        to be easier comparable (partially ignoring PEP8).
        """
        super(HfPooling, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = Sequential(
            Conv2d(1, 20, kernel_size=5),
            ReLU(),
            MaxPool2d(2, stride=2),
            Conv2d(20, 50, kernel_size=5),
            ReLU(),
            MaxPool2d(2, stride=2)
        )
        self.feature_extractor_part2 = Sequential(
            Linear(50 * 4 * 4, self.L),
            ReLU(),
        )

        if mode == 'standard':

            self.hopfield_pooling = HopfieldLayer(
                input_size=self.L, hidden_size=32, output_size=self.L, num_heads=1
            )

        elif mode == 'sparse':
            self.hopfield_pooling = SparseHopfieldLayer(
                input_size=self.L, hidden_size=32, output_size=self.L, num_heads=1
            )

        self.dp = Dropout(
            p=0.1
        )
        self.classifier = Sequential(
            Linear(self.L * self.K, 1),
            Sigmoid()
        )
        
    def forward(self, input: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute result of Hopfield-based pooling network on specified data.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :return: result as computed by the Hopfield-based pooling network
        """
        # x = input.squeeze(0)
        if input.dim() == 5:
            # batched input
            bag_size = input.size(1)
            batch_size = input.size(0)
            c, w, h = input.size(2), input.size(3), input.size(4)
            x = input.view(bag_size*batch_size, c, w, h)
            H = self.feature_extractor_part1(x)
        H = H.view(batch_size, bag_size, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        
        H = self.hopfield_pooling(H, stored_pattern_padding_mask=mask)
        H = self.dp(H)

        Y_prob = self.classifier(H)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, None

    def calculate_classification_error(self, input: Tensor, mask: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute classification error of current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the classification error of the current model
        :return: classification error as well as predicted class
        """
        Y = target.float()
        _, Y_hat, _ = self.forward(input, mask)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, input: Tensor, mask: Tensor, target: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute objective of the current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the objective of the current model
        :return: objective as well as dummy A (see accompanying paper for more information)
        """
        Y = target.float()
        Y_prob, _, A = self.forward(input, mask)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=(1.0 - 1e-5))
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))

        return neg_log_likelihood.mean(), A

def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader
               ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies = [], [], []
    for data, mask, target in tqdm(data_loader):
        data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        loss = network.calculate_objective(data, mask, target)[0]

        # Update network parameters.
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        error, prediction = network.calculate_classification_error(data, mask, target)
        accuracy = (prediction == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        errors.append(error)
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)

def eval_iter(network: Module,
              data_loader: DataLoader
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, errors, accuracies = [], [], []
        for data, mask, target in data_loader:
            data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            loss = network.calculate_objective(data, mask, target)[0]

            # Compute performance measures of current model.
            error, prediction = network.calculate_classification_error(data, mask, target)
            accuracy = (prediction == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            errors.append(error)
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)
   
def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1,
            note = 'none'
           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, errors, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    losses, errors, accuracies = {note: []}, {note: []}, {note: []}
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        # losses[r'train'].append(performance[0])
        # errors[r'train'].append(performance[1])
        # accuracies[r'train'].append(performance[2])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval)
        losses[note].append(performance[0])
        errors[note].append(performance[1])
        accuracies[note].append(performance[2])
    
    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(errors), pd.DataFrame(accuracies)

def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, cpus_per_trial, gpus_per_trial, num_samples=1, max_num_epochs=1):
    
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

    # bag_size = [10, 50, 100, 200, 500, 1000, 2000]
    bag_size = [5, 10, 20, 30, 50]
    # bag_size = [50]

    losss, accs, aucs = [], [], []
    for i, _bag_size in enumerate(bag_size):

        trainset = MnistBags(target_number=9, mean_bag_length=_bag_size, var_bag_length=1, num_bag=1000, seed=args.rs, train=True)
        testset = MnistBags(target_number=9, mean_bag_length=_bag_size, var_bag_length=1, num_bag=1000, seed=args.rs, train=False)

        data_loader_train = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=trainset.collate)
        data_loader_eval = DataLoader(testset, batch_size=4, shuffle=True, collate_fn=testset.collate)
        set_seed(args.rs)
        network = HfPooling(args.mode).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=5e-4, weight_decay=1e-4)

        loss, error, acc = operate(network, optimiser, data_loader_train, data_loader_eval, 20, note=f'bag size {_bag_size}')
        
        loss.to_csv(f'exps/loss_bag{_bag_size}-{args.mode}.csv', index=False)
        acc.to_csv(f'exps/acc_bag{_bag_size}-{args.mode}.csv', index=False)

        losss.append(loss)
        accs.append(acc)

    # loss_plot 
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    # colors = ['red', 'blue', 'green', 'orange', 'brown', 'purple', 'lime']

    all_df = pd.concat(losss, axis=0, ignore_index=True)   
    loss_plot = sns.lineplot(data=all_df, ax=ax[0], hue="coherence")
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Loss', color=colors[i])

    all_df = pd.concat(accs, axis=0, ignore_index=True)   
    loss_plot = sns.lineplot(data=all_df, ax=ax[1], hue="coherence")
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Acc', color=colors[i])

    fig.tight_layout()
    fig.savefig(f'./mnist_performance_{args.mode}.png')

    # print(f"dataset:{args.dataset} loss:{sum(losss)/len(losss)} accs:{sum(accs)/len(accs)} loss:{sum(aucs)/len(aucs)}")



if __name__ == '__main__':
    args = get_args()
    if args.gpus_per_trial>0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_id
    main(args, num_samples=1, max_num_epochs=1, cpus_per_trial=args.cpus_per_trial
         , gpus_per_trial=args.gpus_per_trial)


