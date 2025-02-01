import os
import time
import random

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def train_fn(config):
    """A dummy training function that reports a 'loss' metric."""
    for i in range(10):
        # Simulate some training: a loss function with noise.
        loss = (config["a"] - 7.0) ** 2 + (config["b"] - 5e-4) ** 2 + random.random() * 0.1
        # Report the loss (make sure to pass a dict to avoid keyword issues)
        tune.report({"loss": loss})
        time.sleep(0.1)


if __name__ == "__main__":
    ray.init()

    # Define the search space using Optuna distributions.
    search_space = {
        "a": optuna.distributions.FloatDistribution(6.0, 8.0),
        "b": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
    }

    # Set up external persistent storage for Optuna.
    storage_file_path = "/tmp/optuna_journal.log"
    if os.path.exists(storage_file_path):
        os.remove(storage_file_path)
    storage = JournalStorage(JournalFileBackend(file_path=storage_file_path))

    # Create the OptunaSearch object.
    optuna_search = OptunaSearch(
        space=search_space,
        metric="loss",
        mode="min",
        storage=storage,
        study_name="optuna_parallel_trial",
    )
    # Allow up to 4 concurrent trials at the searcher level.
    optuna_search.set_max_concurrency(4)

    # Wrap the training function to request 1 CPU per trial.
    trainable_with_resources = tune.with_resources(train_fn, {"cpu": 1})

    # Create the Tuner with TuneConfig that allows parallel trials.
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=20,              # Total number of trials.
            max_concurrent_trials=4,     # Tune will schedule up to 4 concurrently.
        ),
        run_config=tune.RunConfig(
            name="optuna_parallel_experiment",
            storage_path="/tmp/tune_results",  # (Optional) for persistent experiment results.
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="loss", mode="min")
    print("Best hyperparameter configuration found:", best_result.config)

    ray.shutdown()
