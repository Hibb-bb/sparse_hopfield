import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from hflayers import *
from sparse_hflayers import *
from datasets.loader import get_dataset, load_ucsb

from sklearn.metrics import roc_auc_score,accuracy_score

import ray
from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from deeprc.dataset_readers import make_dataloaders
from deeprc.predefined_datasets import *


def obtain_loader(args, config):
    if args.dataset == 'cmv':
        task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_dataset(dataset_path='./datasets/cmv/', batch_size=config["batch_size"])

    elif args.dataset == 'implant':
        task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_implanted_dataset(dataset_path='./datasets/cmv_implanted/', batch_size=config["batch_size"], dataset_id=args.dataset_id)

    elif args.dataset == 'lstm':
        task_definition, train_loader, train_loader_fixed, val_loader, test_loader = lstm_generated_dataset(dataset_path='./datasets/lstm/', batch_size=config["batch_size"], dataset_id=args.dataset_id)

    elif args.dataset == 'simulated':
        task_definition, train_loader, train_loader_fixed, val_loader, test_loader = simulated_dataset(dataset_path='./datasets/simulated/', batch_size=config["batch_size"], dataset_id=args.dataset_id)
    
    del task_definition
    del train_loader_fixed

    return train_loader, val_loader, test_loader

def get_args():

    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--name', type=str)
    parser.add_argument('--dataset', default='cmv', type=str, choices=['cmv', 'implant', 'simulated', 'lstm'])
    parser.add_argument('--mode', default='standard', type=str, choices=['standard', 'sparse'])
    parser.add_argument('--rs', help='random state', default=1111, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)
    parser.add_argument('--dataset_id', type=int, default=0)

    args = parser.parse_args()
    return args

class HopfieldMIL(nn.Module):
    def __init__(self, config, feat_dim, mode = 'standard'):
        super(HopfieldMIL, self).__init__()
        emb = [nn.Linear(feat_dim, config["emb_dims"]), nn.ReLU()]
        for i in range(config["emb_layers"] - 1):
            emb.append(nn.Linear(config["emb_dims"], config["emb_dims"]))
            emb.append(nn.ReLU())
        self.emb = nn.ModuleList(emb)

        if mode == 'standard':
            self.hopfield_pooling = HopfieldPooling(
                input_size=config["emb_dims"], num_heads=config["num_heads"], hidden_size = config["hid_dim"],
                scaling=config["scaling_factor"], dropout=config["dropout"]
            )
        elif mode == 'sparse':
            self.hopfield_pooling = SparseHopfieldPooling(
                input_size=config["emb_dims"], num_heads=config["num_heads"], hidden_size = config["hid_dim"],
                scaling=config["scaling_factor"], dropout=config["dropout"]
            )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["emb_dims"], 1)
        )


    def forward(self, x, mask=None):

        H = x.float()
        for l in self.emb:
            H = l(H)
        H = self.hopfield_pooling(H, stored_pattern_padding_mask=mask)
        Y_prob = self.classifier(H).flatten()

        return Y_prob

def train_epoch(network: Module,
                optimizer: torch.optim.AdamW,
                data_loader: DataLoader,
                device
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

    for data, target, mask in data_loader:
        
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
        # if len(losses) % 1 == 0:
        #     p_bar.set_description(f'| Train | Epoch {epoch} | Loss {np.mean(losses)} | Acc {np.mean(accuracies)}')
    
    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def eval_iter(network: Module,
              data_loader: DataLoader,
              device
             ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    # p_bar = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        losses, errors, accuracies, rocs, probs, labels = [], [], [], [], [], []
        for data, target, mask in data_loader:
            
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

            # if len(losses) % 1 == 0 and len(losses) != 0:
            #     p_bar.set_description(f'| Test | Epoch {epoch} | Acc {np.mean(accuracies)}|Roc {np.mean(rocs)}')

        # roc = roc_auc_score(labels, probs)
        # print('ROC', roc)
        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies), sum(rocs)/len(rocs)

def train(config, args, trainset, testset):

    trainloader, valloader, _ = obtain_loader(args.config)

    net = HopfieldMIL(config, feat_dim=args.feat_dim, mode=args.mode)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.AdamW(params=net.parameters(), lr=config['lr'], weight_decay=1e-4)

    # # To restore a checkpoint, use `session.get_checkpoint()`.
    # loaded_checkpoint = session.get_checkpoint()
    # if loaded_checkpoint:
    #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #        model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
    #     net.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    best_val_loss = 1000
    for epoch in range(100):  # loop over the dataset multiple times
        epoch_steps = 0
        _ = train_epoch(net, optimizer, trainloader, device)
        epoch_steps += 1
        for g in optimizer.param_groups:
            g['lr'] *= config["lr_decay"]
        if epoch % 10==0:
            # Validation loss
            val_loss, val_acc, val_auc = eval_iter(net, valloader, device)
            os.makedirs("my_model", exist_ok=True)
            torch.save(
                (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("my_model")
            session.report({"loss": val_loss, "accuracy": val_acc, "auc": val_auc}, checkpoint=checkpoint)

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and can be accessed through `session.get_checkpoint()`
            # API in future iterations.
    
    print("Finished Training")

def test_best_model(best_result, args, testloader):

    best_trained_model = HopfieldMIL(best_result.config, feat_dim=args.feat_dim, mode=args.mode)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_loss,test_acc, test_auc = eval_iter(best_trained_model, testloader, device)

    print("Best trial test set loss: {} acc: {} auc: {}".format(test_loss, test_acc, test_auc))


def main(args, num_samples=10, max_num_epochs=10, gpus_per_trial=2):

    if args.dataset == 'cmv':
        args.datapath = './datasets/cmv/'
    elif args.dataset == 'implant':
        args.datapath = './datasets/cmv_implanted/'
    elif args.dataset == 'simulated':
        args.datapath = './datasets/simulated/'
    elif args.dataset == 'lstm':
        args.datapath = './datasets/lstm/'

    ray.init(num_gpus=3)
    dataset = get_dataset(args, args.dataset)
    trainset = dataset.return_training_set()
    testset = dataset.return_testing_set()
    args.feat_dim = trainset.x[0].shape[-1]

    config = {
        "lr": tune.grid_search([1e-3, 1e-5]),
        "lr_decay": tune.grid_search([0.98,0.96,0.94]),
        "batch_size": tune.grid_search([4,8,16]),
        "emb_dims": tune.grid_search([32, 64, 128]),
        "emb_layers": tune.grid_search([1, 2, 3]),
        "hid_dim": tune.grid_search([16, 32]),
        "num_heads": tune.grid_search([8, 12]),
        "scaling_factor": tune.grid_search([0.1, 1.0, 10.0]),
        "dropout": tune.grid_search([0.0,0.75])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, args=args, trainset=trainset, testset=testset),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="auc",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(local_dir="./results", name=f"{args.mode}_{args.dataset}_{args.name}_noval")
    )
    results = tuner.fit()

    best_result = results.get_best_result("auc", "max")

    trainloader, valloader, testloader = obtain_loader(args.config)
    test_best_model(best_result, args=args, testloader=testloader)

    # print("Best trial config: {}".format(best_result.config))
    # print("Best trial final loss: {}".format(
    #     best_result.metrics["loss"]))
    # print("Best trial final accuracy: {}".format(
    #     best_result.metrics["accuracy"]))
    # print("Best trial final roc-auc: {}".format(
    #     best_result.metrics["auc"]))

    # logs = {
    #         "loss":best_result.metrics["loss"],
    #         "accuracy":best_result.metrics["accuracy"],
    #         "auc":best_result.metrics["auc"]
    #         }
    # test_best_model(best_result, args=args, testset=testset)


if __name__ == '__main__':
    args = get_args()
    main(args, num_samples=1, max_num_epochs=10, gpus_per_trial=0.5)


