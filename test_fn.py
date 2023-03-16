from datasets import loader
import argparse
import torch

parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist'])
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(loader.MnistBags(target_number=9,
                                                mean_bag_length=10,
                                                var_bag_length=2,
                                                num_bag=100,
                                                seed=98,
                                                train=True),
                                                batch_size=4,
                                                shuffle=False)

print(train_loader[0].size())
