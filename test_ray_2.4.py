import os
import ray
from ray import tune
from ray import train
from ray.tune.search.optuna import OptunaSearch
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
import time

# Define the objective function as a Ray remote function
@ray.remote
class Worker:
    def __init__(self):
        self.rng = np.random.RandomState()
    
    def evaluate(self, config):
        # Simulate some computation time
        time.sleep(0.5)  
        x = config["x"]
        y = config["y"]
        noise = self.rng.normal(0, 0.1)
        result = (x - 3)**2 + (y + 4)**2 + noise
        return {"loss": result}

# Trainable class that leverages Ray's distributed execution
class DistributedObjective(tune.Trainable):
    def setup(self, config):
        self.worker = Worker.remote()
        self.config = config
    
    def step(self):
        # This will be executed in parallel across different Ray workers
        result = ray.get(self.worker.evaluate.remote(self.config))
        return result

def main():
    # Initialize Ray with proper resources and enable monitoring
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_port=8465
    )
    
    # Get available CPUs for parallel execution
    num_cpus = ray.cluster_resources().get("CPU", 1)
    max_concurrent = int(num_cpus * 4)  # 4 trials per CPU with 0.25 CPU per trial
    
    # Configure storage path for Optuna
    storage_path = "/tmp/optuna_parallel_storage.log"
    
    # Create Optuna storage with JournalFileBackend
    storage = JournalStorage(
        JournalFileBackend(storage_path)
    )
    
    # Define the search space
    search_space = {
        "x": optuna.distributions.UniformDistribution(-10, 10),
        "y": optuna.distributions.UniformDistribution(-10, 10)
    }
    
    # Initialize OptunaSearch with proper storage
    optuna_search = OptunaSearch(
        search_space,
        storage=storage,
        metric="loss",
        mode="min",
        study_name="parallel_optimization"
    )
    
    # Wrap the trainable with resource specification
    trainable_with_resources = tune.with_resources(
        DistributedObjective,
        {"cpu": 0.25}  # This allows for 4x number of CPU cores parallel trials
    )
    
    # Configure the tuner with explicit parallelism settings
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=100,  # Total number of trials
            max_concurrent_trials=max_concurrent,  # Set based on available CPUs
        ),
        run_config=train.RunConfig(
            name="optuna_parallel_experiment",
            storage_path="/tmp/ray_results",
            sync_config=train.SyncConfig(
                sync_period=60,
                sync_timeout=600,
                sync_artifacts=True,
                sync_artifacts_on_checkpoint=True
            )
        ),
        param_space=search_space
    )
    
    # Run the parallel optimization
    print("\nStarting parallel optimization...")
    print(f"Number of CPUs available: {num_cpus}")
    print(f"Maximum concurrent trials: {max_concurrent}")
    print("Access the Ray dashboard at http://localhost:8265")
    
    results = tuner.fit()
    
    # Get the best result
    best_result = results.get_best_result(metric="loss", mode="min")
    print("\nBest hyperparameters found:", best_result.config)
    print("Best loss:", best_result.metrics["loss"])
    
    # Print parallelism statistics
    print("\nParallel execution statistics:")
    print(f"Number of trials executed: {len(results)}")
    print(f"Number of CPUs used: {num_cpus}")

if __name__ == "__main__":
    main()