import argparse
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from text_utils import *
from tqdm import tqdm
import os
import pandas as pd
from hflayers import HopfieldLayer, HopfieldPooling
from sparse_hflayers import SparseHopfieldLayer, SparseHopfieldPooling
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Module

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from tqdm import tqdm

stop_words = set(stopwords.words('english'))
from d2l import torch as d2l

device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')


def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in tqdm(f.readlines()[1:], desc='extracting file')]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    text = []
    for i in range(len(premises)):
        t = '<cls> ' + premises[i] + '<sep>' + hypotheses[i]
        text.append(t)

    return text, labels

def get_snli(rs=1111):
    data_dir = d2l.download_extract('SNLI')
    train_text, train_label = read_snli(data_dir, is_train=True)

    test_text, test_label = read_snli(data_dir, is_train=False)

    # train_text, train_label, test_text, test_label = get_snli()
    clean_train = [data_preprocessing(t, True) for t in tqdm(train_text, desc='preprocessing train text')]
    clean_test = [data_preprocessing(t, True) for t in tqdm(test_text, desc='preprocessing test text')]
    clean_train, clean_val, train_label, val_label = train_test_split(
            clean_train, train_label, test_size=0.1, random_state=rs)
    vocab = create_vocab(clean_train)

    return clean_train, train_label, clean_val, val_label, clean_test, test_label, vocab

def get_data(config, trainset, valset, testset):

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], collate_fn=trainset.collate, shuffle=True)
    val_loader = DataLoader(
        valset, batch_size=64, collate_fn=valset.collate)
    test_loader = DataLoader(
        testset, batch_size=64, collate_fn=testset.collate)

    return train_loader, val_loader, test_loader

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_auc = 0

    def early_stop(self, validation_auc):
        if validation_auc > self.max_validation_auc:
            self.max_validation_loss = validation_auc
            self.counter = 0
        elif validation_auc < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class TextHopfield(nn.Module):
    def __init__(self, config, mode, word_vec):
        super().__init__()

        self.emb = nn.Embedding.from_pretrained(word_vec, freeze=True)

        if mode == 'dense':
            self.hopfield = HopfieldPooling(
                input_size=config["input_dim"], # embedding size 
                num_heads=config["num_heads"], 
                hidden_size = config["hid_dim"], 
                scaling=config["scaling_factor"], 
                dropout=config["dropout"]
                )
        elif mode == 'sparse':
            self.hopfield = SparseHopfieldPooling(
                input_size=config["input_dim"], 
                num_heads=config["num_heads"], 
                hidden_size = config["hid_dim"], 
                scaling=config["scaling_factor"], 
                dropout=config["dropout"]
                )

        self.fc = nn.LazyLinear(3)

    def forward(self, input, mask):
        x = input
        emb_x = self.emb(x)
        h = self.hopfield(emb_x, stored_pattern_padding_mask=mask)
        return self.fc(h)

def train_epoch(network: Module,
                optimiser: torch.optim.AdamW,
                data_loader: DataLoader
               ):
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        data, mask, target = sample_data
        data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        model_output = network.forward(input=data, mask=mask)

        # Update network parameters.
        optimiser.zero_grad()
        loss = F.cross_entropy(input=model_output, target=target, reduction=r'mean')
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        accuracy = (model_output.argmax(-1) == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

def eval_iter(network: Module,
              data_loader: DataLoader
             ):
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for sample_data in data_loader:
            data, mask, target = sample_data
            data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=data, mask=mask)
            loss = F.cross_entropy(input=model_output, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            accuracy = (model_output.argmax(-1) == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))

def operate(network: Module,
            optimiser: torch.optim.AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            data_loader_valid: DataLoader,
            num_epochs: int = 1,
            note: str = '',
            lr_decay = 0.5
           ):

    losses, accuracies = {f'train': [], r'eval': []}, {r'train': [], r'eval': []}
    best_val_acc = -1
    early_stopper = EarlyStopper()
    network = network.to(device)
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

        if early_stopper.early_stop(performance[1]):
            break

        for g in optimiser.param_groups:
            g['lr'] *= lr_decay

    return best_acc

def single_run(config, mode, word_vec, trainset, valset, testset, tune=False):

    model = TextHopfield(config, mode, word_vec)
    opt = torch.optim.AdamW(model.parameters(), lr = config["lr"])

    train_loader, val_loader, test_loader = get_data(config, trainset, valset, testset)

    best_acc = operate(model, opt, train_loader, val_loader, test_loader, num_epochs=20, lr_decay=config["lr_decay"])

    if tune:
        session.report({"acc", best_acc})
    else:
        return best_acc

def tune_config(word_vec, trainset, valset, testset, mode='dense'):

    # config = {
    #     "lr": tune.grid_search([1e-3, 1e-5]),
    #     "lr_decay": tune.grid_search([0.98, 0.96, 0.94]),
    #     "batch_size": tune.grid_search([16, 32, 64]),
    #     "hid_dim": tune.grid_search([32, 64]),
    #     "num_heads": tune.grid_search([8, 12]),
    #     "scaling_factor": tune.grid_search([0.1, 10.0]),
    #     "dropout": tune.grid_search([0.0, 0.75]),
    #     "input_dim": tune.grid_search([300])
    # }

    config = {
        "lr": tune.grid_search([1e-3]),
        "lr_decay": tune.grid_search([0.98]),
        "batch_size": tune.grid_search([64]),
        "hid_dim": tune.grid_search([64]),
        "num_heads": tune.grid_search([12]),
        "scaling_factor": tune.grid_search([0.1]),
        "dropout": tune.grid_search([0.0]),
        "input_dim": tune.grid_search([300])
    }

    scheduler = ASHAScheduler(
        max_t=1,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(single_run, mode=mode, word_vec=word_vec, trainset=trainset
                                    , valset=valset, testset=testset, tune=True),
            resources={"cpu": 1, "gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            metric="acc",
            mode="max",
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=config,
        run_config=RunConfig(local_dir="./results"
                                , name=f"{mode}_SNLI/")
    )
    results = tuner.fit()

    best_result = results.get_best_result("acc", "max")

    best_config = best_result.config
    best_acc = best_result.metrics["acc"]

    return best_config, best_acc

def run_exp():

    MAX_LEN=200
    clean_train, train_label, clean_val, val_label, clean_test, test_label, vocab = get_snli()
    word_vec = get_word_vector(vocab)

    trainset = Textset(clean_train, train_label, vocab, MAX_LEN)
    valset = Textset(clean_val, val_label, vocab, MAX_LEN)
    testset = Textset(clean_test, test_label, vocab, MAX_LEN)

    torch.manual_seed(1111)

    results = {'dense':[], 'sparse':[], 'attn':[]}
    dense_config, dense_acc = tune_config(word_vec=word_vec, mode='dense', trainset=trainset, valset=valset, testset=testset)
    sparse_config, sparse_acc = tune_config(word_vec= word_vec, mode='sparse', trainset=trainset, valset=valset, testset=testset)
    # attn_config, attn_acc = tune_config(mode='attn', num_bag=n)

    results["dense"].append(dense_acc)
    results["sparse"].append(sparse_acc)
    # results["attn"].append(attn_acc)

    for i in range(2, 3):

        rs = int(i*1111)
        torch.manual_seed(rs)
        dense_acc = single_run(config=dense_config, word_vec=word_vec, mode='dense', trainset=trainset, valset=valset, testset=testset, tune=False)
        sparse_acc = single_run(config=sparse_config, word_vec=word_vec, trainset=trainset, valset=valset, testset=testset, tune=False)
        # attn_acc = single_run(config=attn_config,  trainset=trainset, valset=valset, testset=testset, tune=False)

        results["dense"].append(round(dense_acc, 4))
        results["sparse"].append(round(sparse_acc,4))
        # results["attn"].append(round(attn_acc, 4))

    table = [['dense', 'sparse']]

    tab = PrettyTable(table[0])
    # r.append(f"{round(lstm_res[0][i], 4)}, {round(lstm_res[1][i], 2)}")
    r.append(f"{np.mean(results['dense'])}, {np.std(results['dense'])}")
    r.append(f"{np.mean(results['sparse'])}, {np.std(results['sparse'])}")
    tab.add_rows([r])

    print(tab)


if __name__ == '__main__':
    run_exp()