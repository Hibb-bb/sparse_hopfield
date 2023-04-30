import argparse
from ray import tune
import ray

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print results')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    experiment_path = args.path
    print(f"Loading results from {experiment_path}...")

    ray.init()

    restored_tuner = tune.Tuner.restore(experiment_path)
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result(
        metric="auc", mode="max")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final validation roc-auc: {}".format(
        best_result.metrics["auc"]))
    # print("Best trial final test loss: {}".format(
    #     best_result.metrics["test_loss"]))
    # print("Best trial final test accuracy: {}".format(
    #     best_result.metrics["test_accuracy"]))
    # print("Best trial final test roc-auc: {}".format(
    #     best_result.metrics["test_auc"]))
