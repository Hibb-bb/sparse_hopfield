import nltk
nltk.download('stopwords')
import argparse
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from text_utils2 import *
from tqdm import tqdm
import os
import pandas as pd
from hflayers import HopfieldLayer, Hopfield
# from hflayers.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from sparse_hflayers import SparseHopfieldLayer, SparseHopfield
# from sparse_hflayers.transformer import SparseHopfieldEncoderLayer, SparseHopfieldDecoderLayer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Module
from prettytable import PrettyTable

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from ray import tune
from ray.air import session, RunConfig
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from tqdm import tqdm

stop_words = set(stopwords.words('english'))
from d2l import torch as d2l

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
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    # text = []
    # for i in range(len(premises)):
    #     t = '<cls> ' + premises[i] + '<sep>' + hypotheses[i]
    #     text.append(t)

    return premises, hypotheses, labels

def get_snli(rs=1111):
    data_dir = d2l.download_extract('SNLI')
    train_p, train_h, train_label = read_snli(data_dir, is_train=True)
    test_p, test_h, test_label = read_snli(data_dir, is_train=False)

    # train_text, train_label, test_text, test_label = get_snli()
    clean_train_p = [data_preprocessing(t, True) for t in train_p]
    clean_train_h = [data_preprocessing(t, True) for t in train_h]

    clean_test_p = [data_preprocessing(t, True) for t in test_p]
    clean_test_h = [data_preprocessing(t, True) for t in test_h]

    clean_train_p, clean_val_p, clean_train_h, clean_val_h, train_label, val_label = train_test_split(
            clean_train_p, clean_train_h, train_label, test_size=0.1, random_state=rs)
    vocab = create_vocab(clean_train_p + clean_train_h)

    return clean_train_p, clean_train_h, train_label, clean_val_p, clean_val_h, val_label, clean_test_p, clean_test_h, test_label, vocab

def get_data(config, trainset, valset, testset):

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], collate_fn=trainset.collate, shuffle=True)
    val_loader = DataLoader(
        valset, batch_size=config["batch_size"], collate_fn=valset.collate)
    test_loader = DataLoader(
        testset, batch_size=config["batch_size"], collate_fn=testset.collate)

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
        elif validation_auc < (self.max_validation_auc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class TextHopfield(nn.Module):
    def __init__(self, config, mode, word_vec):
        super().__init__()

        self.emb = nn.Embedding.from_pretrained(word_vec, freeze=False)

        if mode == 'dense':
            self.enc = Hopfield(
                        input_size = config["input_dim"],
                        num_heads=config["num_heads"], 
                        hidden_size = config["hid_dim"], 
                        scaling=config["scaling_factor"], 
                        dropout=config["dropout"],
                        update_steps_max=3
                        )

            self.dec = Hopfield(
                        input_size = config["input_dim"],
                        num_heads=config["num_heads"], 
                        hidden_size = config["hid_dim"], 
                        scaling=config["scaling_factor"], 
                        dropout=config["dropout"],
                        update_steps_max=3
                        )

        elif mode == 'sparse':
            self.enc = SparseHopfield(
                        input_size = config["input_dim"],
                        num_heads=config["num_heads"], 
                        hidden_size = config["hid_dim"], 
                        scaling=config["scaling_factor"], 
                        dropout=config["dropout"],
                        update_steps_max=3
                        )

            self.dec = SparseHopfield(
                        input_size = config["input_dim"],
                        num_heads=config["num_heads"], 
                        hidden_size = config["hid_dim"], 
                        scaling=config["scaling_factor"], 
                        dropout=config["dropout"],
                        update_steps_max=3
                        )
        self.fc = nn.Linear(config["input_dim"], 3)

    def forward(self, x1, x2, mask1, mask2):

        emb_x1 = self.enc(self.emb(x1), stored_pattern_padding_mask=mask1)
        emb_x2 = self.enc(self.emb(x2), stored_pattern_padding_mask=mask2)

        h = self.dec( input=(emb_x1, emb_x2, emb_x1), stored_pattern_padding_mask=mask1)
        h = h.reshape(x1.size(0), -1, emb_x1.size(-1)) # (batch_size, len, hid)
        # print(h.size())
        h = h.mean(1)
        return self.fc(h)

def train_epoch(network: Module,
                optimiser: torch.optim.AdamW,
                data_loader: DataLoader
               ):
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        data, mask, data2, mask2, target = sample_data
        data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)
        data2, mask2 = data2.to(device=device), mask2.to(device=device)
        
        # Process data by Hopfield-based network.
        model_output = network.forward(data, data2, mask, mask2)
        # Update network parameters.
        optimiser.zero_grad()
        loss = F.cross_entropy(input=model_output, target=target, reduction=r'mean')
        print('loss', loss.item())
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

            data, mask, data2, mask2, target = sample_data
            data, mask, target = data.to(device=device), mask.to(device=device), target.to(device=device)
            data2, mask2 = data2.to(device=device), mask2.to(device=device)
            
            # Process data by Hopfield-based network.
            model_output = network.forward(data, data2, mask, mask2)

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
        print('train', performance)
        # Evaluate current model.
        performance = eval_iter(network, data_loader_valid)
        print('eval', performance)

        if performance[1] >= best_val_acc:
            best_val_acc = performance[1]
            p = eval_iter(network, data_loader_eval)
            best_acc = p[1]
        # print(performance)
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])

        if early_stopper.early_stop(performance[0]):
            break

        for g in optimiser.param_groups:
            g['lr'] *= lr_decay
    # print('best', best_acc)
    return best_acc

def single_run(config, mode, word_vec, trainset, valset, testset, tune=False):

    model = TextHopfield(config, mode, word_vec)
    opt = torch.optim.AdamW(model.parameters(), lr = config["lr"])

    train_loader, val_loader, test_loader = get_data(config, trainset, valset, testset)

    best_acc = operate(model, opt, train_loader, val_loader, test_loader, num_epochs=50, lr_decay=config["lr_decay"])

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
        "num_heads": tune.grid_search([1]),
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
            resources={"cpu": 1, "gpu": 0.25}
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
    clean_train_p, clean_train_h, train_label, clean_val_p, clean_val_h, val_label, clean_test_p, clean_test_h, test_label, vocab = get_snli()
    word_vec = get_word_vector(vocab)

    trainset = Textset(clean_train_p, clean_train_h, train_label, vocab, MAX_LEN)
    valset = Textset(clean_val_p, clean_val_h, val_label, vocab, MAX_LEN)
    testset = Textset(clean_test_p, clean_test_h, test_label, vocab, MAX_LEN)

    config = {
        "lr": 1e-5,
        "lr_decay": 0.98,
        "batch_size": 512,
        "hid_dim": 64,
        "num_heads": 8,
        "scaling_factor": 0.1,
        "dropout":0.0,
        "input_dim": 300
    }
    dense_config = config
    sparse_config = config

    torch.manual_seed(1111)

    results = {'dense':[], 'sparse':[], 'attn':[]}
    # dense_config, dense_acc = tune_config(word_vec=word_vec, mode='dense', trainset=trainset, valset=valset, testset=testset)
    # sparse_config, sparse_acc = tune_config(word_vec= word_vec, mode='sparse', trainset=trainset, valset=valset, testset=testset)
    # attn_config, attn_acc = tune_config(mode='attn', num_bag=n)

    # results["dense"].append(dense_acc)
    # results["sparse"].append(sparse_acc)
    # results["attn"].append(attn_acc)

    for i in range(2, 3):

        rs = int(i*1111)
        torch.manual_seed(rs)
        dense_acc = single_run(config=dense_config, word_vec=word_vec, mode='dense', trainset=trainset, valset=valset, testset=testset, tune=False)
        print(dense_acc)
        
        sparse_acc = single_run(config=sparse_config, word_vec=word_vec, mode='sparse',  trainset=trainset, valset=valset, testset=testset, tune=False)
        print(sparse_acc)

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
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
    run_exp()
