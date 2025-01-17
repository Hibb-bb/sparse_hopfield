from datasets import loader

# MnistBags
import argparse

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='ucsb', type=str, choices=['ucsb'])
parser.add_argument('--rs', help='random state', default=1111, type=int)
parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

dataset = loader.MnistBags()