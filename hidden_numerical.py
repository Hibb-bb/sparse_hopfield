# Import general modules used e.g. for plotting.
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import torch

# Importing Hopfield-specific modules.
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
from hflayers.auxiliary.data import BitPatternSet
from sparse_hflayers import SparseHopfield, SparseHopfieldPooling, SparseHopfieldLayer

# Import auxiliary modules.
from distutils.version import LooseVersion
from typing import List, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Flatten, Linear, Module, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Set plotting style.
sns.set()

log_dir = f'resources_hidden/'
os.makedirs(log_dir, exist_ok=True)

device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

def get_data(num_bags=2048, num_instances=16, num_signals=8, num_signals_per_bag=1, num_bits=8):
    
    bit_pattern_set = BitPatternSet(
        num_bags=num_bags,
        num_instances=num_instances,
        num_signals=num_signals,
        num_signals_per_bag=num_signals_per_bag,
        num_bits=num_bits)

    # Create data loader of training set.
    sampler_train = SubsetRandomSampler(list(range(256, 2048 - 256)))
    data_loader_train = DataLoader(dataset=bit_pattern_set, batch_size=128, sampler=sampler_train)

    # Create data loader of validation set.
    sampler_eval = SubsetRandomSampler(list(range(256)) + list(range(2048 - 256, 2048)))
    data_loader_eval = DataLoader(dataset=bit_pattern_set, batch_size=128, sampler=sampler_eval)

    return data_loader_train, data_loader_eval, bit_pattern_set

def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader
               ) -> Tuple[float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        data, target = sample_data[r'data'], sample_data[r'target']
        data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        model_output = network.forward(input=data)

        # Update network parameters.
        optimiser.zero_grad()
        loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

def eval_iter(network: Module,
              data_loader: DataLoader
             ) -> Tuple[float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for sample_data in data_loader:
            data, target = sample_data[r'data'], sample_data[r'target']
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=data)
            loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval)
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
    
    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)

def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_performance(loss: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_file: str
                    ) -> None:
    """
    Plot and save loss and accuracy.
    
    :param loss: loss to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    
    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Cross-entropy Loss')
    
    accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Accuracy')
    
    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    fig.savefig(log_file)
    plt.close()
    # plt.show(fig)

def build_hopfield(mode='standard', hidden_size=8, bit_pattern_set=None):
    if mode == 'standard':
        hopfield = Hopfield(
            input_size=bit_pattern_set.num_bits,
            update_steps_max=3,
            num_heads=8,
            hidden_size=hidden_size,
            scaling=0.25,
            dropout=0.5)
        output_projection = Linear(in_features=hopfield.output_size * bit_pattern_set.num_instances, out_features=1)
        network = Sequential(hopfield, Flatten(), output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
    elif mode == 'sparse':
        hopfield = SparseHopfield(
            input_size=bit_pattern_set.num_bits,
            update_steps_max=3,
            num_heads=8,
            hidden_size=hidden_size,
            scaling=0.25,
            dropout=0.5)
        output_projection = Linear(in_features=hopfield.output_size * bit_pattern_set.num_instances, out_features=1)
        network = Sequential(hopfield, Flatten(), output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
    
    return network, optimiser

def run_hopfield_exp():

    num_instance = [64]
    hidden_size = [8, 16, 32, 64, 128]
    max_acc = []
    for n in hidden_size:

        train_loader, eval_loader, bit_pattern_set = get_data(num_instances=64)
        
        sparse_res, res = 0.0, 0.0
        for i in range(1, 6):
            set_seed(int(i*1000))
            model, opt = build_hopfield('standard', n, bit_pattern_set)
            loss, acc = operate(network=model, optimiser=opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=250)

            sparse_model, sparse_opt = build_hopfield('sparse', n, bit_pattern_set)
            sparse_loss, sparse_acc = operate(network=sparse_model, optimiser=sparse_opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=250)

            sparse_res += max(sparse_acc['eval'])
            res += max(acc['eval'])

            plot_performance(loss=loss, accuracy=acc, log_file=f'{log_dir}/hopfield_base_hid{n}_standard_rs{i}.pdf')
            plot_performance(loss=sparse_loss, accuracy=sparse_acc, log_file=f'{log_dir}/hopfield_base_hid{n}_sparse_rs{i}.pdf')
            
        max_acc.append([res/5, sparse_res/5])

    for i in range(len(hidden_size)):
        print('Instance=', hidden_size[i], 'standard performance', max_acc[i][0], 'sparse performance', max_acc[i][1])

def build_pooling(mode='sparse', bit_pattern_set=None):

    if mode == 'standard':

        hopfield_pooling = HopfieldPooling(
            input_size=bit_pattern_set.num_bits)
        output_projection = Linear(in_features=hopfield_pooling.output_size, out_features=1)
        network = Sequential(hopfield_pooling, output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
        return network, optimiser
    else:
        hopfield_pooling = SparseHopfieldPooling(
            input_size=bit_pattern_set.num_bits)
        output_projection = Linear(in_features=hopfield_pooling.output_size, out_features=1)
        network = Sequential(hopfield_pooling, output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
        return network, optimiser

def run_pooling_exp():

    num_instance = [16, 32, 64, 128, 256]
    max_acc = []
    for n in num_instance:

        train_loader, eval_loader, bit_pattern_set = get_data(num_instances=n)
        model, opt = build_pooling('standard', bit_pattern_set)
        loss, acc = operate(network=model, optimiser=opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=200)

        sparse_model, sparse_opt = build_pooling('sparse', bit_pattern_set)
        sparse_loss, sparse_acc = operate(network=sparse_model, optimiser=sparse_opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=200)

        plot_performance(loss=loss, accuracy=acc, log_file=f'{log_dir}/hopfield_base_{n}_standard.pdf')
        plot_performance(loss=sparse_loss, accuracy=sparse_acc, log_file=f'{log_dir}/hopfield_pooling_{n}_sparse.pdf')

        max_acc.append([max(acc['eval']), max(sparse_acc['eval'])])

    for i in range(len(num_instance)):
        print('[Pooling] Instance=', num_instance[i], 'standard performance', max_acc[i][0], 'sparse performance', max_acc[i][1])

def build_layer(mode, bit_pattern_set, bit_samples_unique):

    set_seed()
    if mode == 'standard':
        hopfield_lookup = HopfieldLayer(
            input_size=bit_pattern_set.num_bits,
            quantity=len(bit_samples_unique))
        output_projection = Linear(in_features=hopfield_lookup.output_size * bit_pattern_set.num_instances, out_features=1)
        network = Sequential(hopfield_lookup, Flatten(start_dim=1), output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
    
    else:
        hopfield_lookup = SparseHopfieldLayer(
            input_size=bit_pattern_set.num_bits,
            quantity=len(bit_samples_unique))
        output_projection = Linear(in_features=hopfield_lookup.output_size * bit_pattern_set.num_instances, out_features=1)
        network = Sequential(hopfield_lookup, Flatten(start_dim=1), output_projection, Flatten(start_dim=0)).to(device=device)
        optimiser = AdamW(params=network.parameters(), lr=1e-3)
    return network, optimiser

def run_layer_exp():

    num_instance = [16, 32, 64, 128, 256]
    max_acc = []
    for n in num_instance:

        train_loader, eval_loader, bit_pattern_set = get_data(num_instances=n)
        bit_samples_unique = [_[r'data'] for _ in data_loader_train]
        bit_samples_unique = torch.cat(bit_samples_unique).view(-1, bit_samples_unique[0].shape[2]).unique(dim=0)

        model, opt = build_layer('standard', bit_pattern_set, bit_samples_unique)
        loss, acc = operate(network=model, optimiser=opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=250)

        sparse_model, sparse_opt = build_layer('sparse', bit_pattern_set, bit_samples_unique)
        sparse_loss, sparse_acc = operate(network=sparse_model, optimiser=sparse_opt, data_loader_train=train_loader, data_loader_eval=eval_loader, num_epochs=250)

        plot_performance(loss=loss, accuracy=acc, log_file=f'{log_dir}/hopfield_base_{n}_standard.pdf')
        plot_performance(loss=sparse_loss, accuracy=sparse_acc, log_file=f'{log_dir}/hopfield_layer_{n}_sparse.pdf')

        max_acc.append([max(acc['eval']), max(sparse_acc['eval'])])


    for i in range(len(num_instance)):
        print('[Layer] Instance=', num_instance[i], 'standard performance', max_acc[i][0], 'sparse performance', max_acc[i][1])

run_hopfield_exp()