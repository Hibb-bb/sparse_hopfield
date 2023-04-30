from ray import tune

if __name__ == '__main__':
    tuner = tune.Tuner.restore(path="~/ray_results/my_experiment")