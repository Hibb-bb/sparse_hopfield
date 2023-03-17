import numpy as np
import scipy.io
import os
import pickle

import pandas as pd
import sklearn.model_selection

import torch
import torch.utils.data
from torchvision import datasets, transforms

from .loader_utils import *


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = self.x[idx] # (bag_size, feat_dim)
        batch_y = self.y[idx]

        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)

        return batch_x, batch_y

    def collate(self, batch):

        x = [x for x,y in batch]
        y = [y for x,y in batch]

        return x, y

def load_ucsb():
    
    '''
    This function Returns trainset and testset
    '''

    def load_data(filepath):
        df = pd.read_csv(filepath, header=None)
        
        bags_id = df[1].unique()
        bags = [df[df[1]==bag_id][df.columns.values[2:]].values.tolist() for bag_id in bags_id]
        y = df.groupby([1])[0].first().values

        # split train and test data
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(bags, y, test_size=0.2, random_state=0)
        
        trainset, testset = DummyDataset(X_train, y_train), DummyDataset(X_test, y_test)

        return trainset, testset

    current_file = os.path.abspath(os.path.dirname(__file__))
    return load_data(current_file + '/csv/ucsb_breast_cancer.csv')

def get_dataset(args, dataset='fox'):
    """
    Loads and batches fox dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    if args.multiply:
        filepath = os.getcwd() + '/datasets/mil_datasets/{}_dataset.pkl'.format(args.dataset)
    else:
        filepath = os.getcwd() + '/datasets/mil_datasets/{}_original_dataset.pkl'.format(args.dataset)
    if (os.path.exists(filepath)):
        print('Dataset loaded')
        with open(filepath, 'rb') as dataset_file:
            dataset =  pickle.load(dataset_file)
            return dataset
    else:
        dataset = Dataset(args, dataset)
        print('Dataset loaded')
        file = open(filepath, 'wb')
        pickle.dump(dataset, file)
        return dataset

class Dataset():
    def __init__(self, args, dataset='fox'):
        """
        Loads and batches elephant dataset into feature and bag label lists
        :return: list(features), list(bag_labels)
        """
        self.rs = args.rs # random state
        self.features = []
        self.bag_labels = []
        dataset = scipy.io.loadmat(os.getcwd() + f'/datasets/mil_datasets/{dataset}_100x100_matlab.mat')  # loads fox dataset
        instance_bag_ids = np.array(dataset['bag_ids'])[0]
        instance_features = np.array(dataset['features'].todense())
        print(instance_features[0].shape)
        if args.multiply:
            instance_features = multiply_features(instance_features)

        instance_labels = np.array(dataset['labels'].todense())[0]
        bag_features = into_dictionary(instance_bag_ids,
                                       instance_features)  # creates dictionary whereas key is bag and values are
        bag_labels = into_dictionary(instance_bag_ids,
                                     instance_labels)  # creates dictionary whereas key is bag and values are instance
        for i in range(1, 201):  # goes through whole dataset
            self.features.append(np.array(bag_features.pop(i)))
            self.bag_labels.append(max(bag_labels[i]))
        self.random_shuffle()

    def random_shuffle(self):
        self.features, self.bag_labels = shuffle_dataset(self.features, self.bag_labels, self.rs)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.bag_labels)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test

    def return_training_set(self):

        trainset = DummyDataset(self.training_data, self.training_labels)
        return trainset

    def return_testing_set(self):
        testset = DummyDataset(self.testing_data, self.testing_labels)
        return testset

    def return_dataset(self):
        fullset = DummyDataset(self.features, self.bag_labels)
        return fullset

class MnistBags(torch.utils.data.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=1, num_bag=1000, seed=7, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.seed = seed
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._form_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._form_bags()

    def _form_bags(self):
        if self.train:
            train_loader = torch.utils.data.DataLoader(datasets.MNIST('../datasets',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))])),
                                                 batch_size=self.num_in_train,
                                                 shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in train_loader:
                numbers = batch_data[0]
                labels = batch_data[1]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        else:
            test_loader = torch.utils.data.DataLoader(datasets.MNIST('../datasets',
                                                               train=False,
                                                               download=True,
                                                               transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=self.num_in_test,
                                                shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in test_loader:
                numbers = batch_data[0]
                labels = batch_data[1]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_test, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label