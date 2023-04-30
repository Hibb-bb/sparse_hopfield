import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hflayers import *
from sparse_hflayers import *
from datasets.loader import get_dataset, load_ucsb

from sklearn.metrics import roc_auc_score



def get_args():

    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default='fox', type=str, choices=['fox', 'tiger', 'elephant'])
    parser.add_argument('--rs', help='random state', default=1111, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

    # model parameters
    parser.add_argument('--feat-dim', default=708, type=int)
    parser.add_argument('--emb-layer', default=1, type=int)
    parser.add_argument('--hid-dim', default=128, type=int)
    parser.add_argument('--mode', default='standard', choices=['standard', 'sparse', 'entmax'])
    parser.add_argument('--num-heads', default=8, type=int)
    # parser.add_argument('--beta', default=10.0, type=float, choices=[0.1, 1.0, 10.0])
    parser.add_argument('--beta', default=10.0, type=float)

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250)

    args = parser.parse_args()
    return args

class HopfieldMIL(nn.Module):
    def __init__(self, args, emb_dims = [256],  mode = 'standard'):
        super().__init__()

        self.L = 128
        self.D = 128
        self.K = 1

        emb = [nn.Linear(args.feat_dim, emb_dims[0]), nn.ReLU()]
        for i in range(args.emb_layer - 1):
            emb.append(nn.Linear(emb_dims[i], emb_dims[i+1]))
            emb.append(nn.ReLU())
        self.emb = nn.ModuleList(emb)
        self.bag_dropout = nn.Dropout2d(p=0.75)

        if mode == 'standard':
            self.hopfield_pooling = HopfieldPooling(
                input_size=emb_dims[-1], num_heads=args.num_heads, hidden_size = args.hid_dim,
                scaling=args.beta, update_steps_max=3, dropout=0.0
            )
        elif mode == 'sparse':
            self.hopfield_pooling = SparseHopfieldPooling(
                input_size=emb_dims[-1], num_heads=args.num_heads,hidden_size = args.hid_dim,
                scaling=args.beta, update_steps_max=3,
            )

        # self.dp = nn.Dropout(
        #     p=0.1
        # )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dims[-1], 1)
        )


    def forward(self, x, mask=None):

        H = x.float()
        for l in self.emb:
            H = l(H)
        # H = self.bag_dropout(H)
        H = self.hopfield_pooling(H, stored_pattern_padding_mask=mask)
        # H = H.squeeze(0)
        # H = self.dp(H)

        Y_prob = self.classifier(H).flatten()
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob

    def calculate_classification_error(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute classification error of current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the classification error of the current model
        :return: classification error as well as predicted class
        """
        Y = target.float()
        _, Y_hat, _ = self.forward(input)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute objective of the current model.
        
        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the objective of the current model
        :return: objective as well as dummy A (see accompanying paper for more information)
        """
        Y = target.float()
        Y_prob, _, A = self.forward(input)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=(1.0 - 1e-5))
        neg_log_likelihood = -1.0 * (Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob))

        return neg_log_likelihood, A

def train_epoch(network: Module,
                optimizer: torch.optim.AdamW,
                data_loader: DataLoader,
                device,
                epoch
               ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies, rocs = [], [], [], []
    p_bar = tqdm(data_loader, total=len(data_loader))

    for data, target, mask in p_bar:
        
        data, target, mask = data.to(device=device), target.to(device=device).float(), mask.to(device)

        # Process data by Hopfield-based network.
        out = network(data, mask=mask)
        
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

        # Update network parameters.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        # Compute performance measures of current model.
        accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        # roc = roc_auc_score(target.squeeze().detach().cpu(), out.sigmoid().squeeze().detach().cpu())
        # rocs.append(roc)
        losses.append(loss.detach().item())
        if len(losses) % 1 == 0: 
            p_bar.set_description(f'| Train | Epoch {epoch} | Loss {np.mean(losses)} | Acc {np.mean(accuracies)}')
    
    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def eval_iter(network: Module,
              data_loader: DataLoader,
              device,
              epoch
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    p_bar = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        losses, errors, accuracies, rocs, probs, labels = [], [], [], [], [], []
        for data, target, mask in p_bar:
            
            data, target, mask = data.to(device=device), target.to(device=device).float(), mask.to(device)

            # Process data by Hopfield-based network
            out = network(data, mask=mask)
            loss = F.binary_cross_entropy_with_logits(input=out, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            probs = probs + (torch.sigmoid(out).squeeze(-1).tolist())
            labels = labels + (target.squeeze(-1).tolist())

            accuracy = (out.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            roc = roc_auc_score(target.squeeze().detach().cpu(), out.sigmoid().squeeze().detach().cpu())
            rocs.append(roc)
            losses.append(loss.detach().item())

            if len(losses) % 1 == 0 and len(losses) != 0: 
                p_bar.set_description(f'| Test | Epoch {epoch} | Acc {np.mean(accuracies)}|Roc {np.mean(rocs)}')

        # roc = roc_auc_score(labels, probs)
        # print('ROC', roc)
        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def operate(network: Module,
            optimizer: torch.optim.AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
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
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_test_acc = -1

    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimizer, data_loader_train, torch.device('cuda:1'), epoch)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval, torch.device('cuda:1'), epoch)
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
        if best_test_acc <= performance[1]:
            best_test_acc = performance[1]

        for g in optimizer.param_groups:
            g['lr'] *= 0.98

        # sch.step()
    print('best acc', best_test_acc)

    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)

if __name__ == '__main__':
    
    args = get_args()

    # trainset, testset = load_ucsb()

    dataset = get_dataset(args, 'fox')
    trainset = dataset.return_training_set()
    testset = dataset.return_testing_set()

    args.feat_dim = trainset.x[0].shape[-1]

    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=trainset.collate)
    test_loader = DataLoader(testset, batch_size=len(testset.x), collate_fn=testset.collate)

    model = HopfieldMIL(args, mode=args.mode)
    model = model.to(device='cuda:1')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    operate(model, optimizer, train_loader, test_loader, args.epochs)