from datasets import loader
import argparse

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='ucsb', type=str, choices=['ucsb'])
parser.add_argument('--rs', help='random state', default=1111, type=int)
parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

args = parser.parse_args()

args = parser.parse_args()

dataset = loader.get_dataset(args, 'fox')
trainset = dataset.return_training_set()

tdl = DataLoader(trainset, batch_size=4, collate_fn=trainset.collate)

trainset, testset = loader.load_ucsb()

tld = DataLoader(trainset, batch_size=2, collate_fn=trainset.collate)

