import ray
from ray import tune
from ray.tune.search import Searcher
import optuna
import numpy as np
import os
import time
from optuna.distributions import UniformDistribution, IntUniformDistribution
from ray.tune.search.sample import Float, Integer

class OptunaRaySearch(Searcher):
    def __init__(self, space, metric, mode, storage, study_name, **kwargs):
        super().__init__(metric=metric, mode=mode, **kwargs)
        self.space = self._convert_space(space)
        self.storage = storage
        self.study_name = study_name
        self._study = None
        self._current_trial = None
        self._initial_search = True
        self._direction = "minimize" if mode == "min" else "maximize"
        self._study = self._get_study()

    def _convert_space(self, space):
        optuna_space = {}
        for key, value in space.items():
            if isinstance(value, Float):
                optuna_space[key] = UniformDistribution(value.lower, value.upper)
            elif isinstance(value, Integer):
                optuna_space[key] = IntUniformDistribution(value.lower, value.upper)
            else:
                raise ValueError(f"Unsupported distribution type: {type(value)}. Please use tune.uniform, tune.quniform or tune.int")
        return optuna_space

    def _get_study(self):
        if self._study is None:
            self._study = optuna.create_study(
                storage=self.storage,
                study_name=self.study_name,
                direction=self._direction,
                load_if_exists=True
            )
        return self._study

    def suggest(self, trial_id):
        study = self._get_study()
        if self._initial_search:
            self._initial_search = False
            # Returns a dict of distributions for initial search
            return {key: value for key, value in self.space.items()}
        
        self._current_trial = study.ask(self.space)
        return self._current_trial.params  # Return sampled values

    def on_trial_complete(self, trial_id, result=None, error=None):
      if error :
        self._get_study().tell(self._current_trial, state = optuna.trial.TrialState.FAIL)
      elif result is not None:
        self._get_study().tell(self._current_trial, result=result[self._metric])
        
      # We also need to inform the tune scheduler about it, so this is required.
      super().on_trial_complete(trial_id, result, error)

def objective(config):
    score = (config["x"] - 2) ** 2 + (config["y"] + 3) ** 2
    return {"score": score}

if __name__ == "__main__":
    ray.init()
    
    # 1. Define Optuna's Storage
    storage_path = "sqlite:///optuna_study.db"  # Path to SQLite DB file
    study_name = "my_study"
    
    # 2. Create an Optuna Search object with persistent storage
    optuna_search = OptunaRaySearch(
        space = {
            "x" : tune.uniform(-5,5),
            "y" : tune.uniform(-5,5),
        },
        metric="score",
        mode="min",
        storage=storage_path,
        study_name=study_name
    )
    
    # 3. Configure Ray Tune
    tuner = tune.Tuner(
        tune.with_resources(objective, resources={"cpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=100,
        ),
        run_config=ray.air.RunConfig(
            name="optuna_parallel_example",
            verbose = 3,
        ),
    )
    results = tuner.fit()
    print("Best hyperparameters: ", results.get_best_result().config)
    ray.shutdown()