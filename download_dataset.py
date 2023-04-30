from deeprc.dataset_readers import make_dataloaders
from deeprc.predefined_datasets import *


batch_size = 4

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_dataset(dataset_path='./datasets/cmv/')

# task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_implanted_dataset(dataset_path='./datasets/cmv_implanted/', batch_size=batch_size)

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = lstm_generated_dataset(dataset_path='./datasets/lstm/', batch_size=batch_size)

# task_definition, trainingset, trainingset_eval, validationset_eval, testset_eval = simulated_dataset()