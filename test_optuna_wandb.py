import optuna
import optunahub
from optuna_integration.wandb import WeightsAndBiasesCallback
import multiprocessing
import wandb

# Define objective function (same)
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_categorical('y', ['linear', 'rbf', 'poly'])
    z = trial.suggest_int('z', 1, 10)
    score = (x - 2) ** 2 + (z-5)**2
    if y == 'linear':
        score += 1
    elif y == 'rbf':
        score += 2
    else:
        score += 3
    trial.report(score, step=1) # Keep step=1
    return score

def run_optimization_process(wandbc_callback): # Pass the callback instance
    module = optunahub.load_module(package="samplers/auto_sampler")
    auto_sampler = module.AutoSampler()

    study = optuna.create_study(
        study_name="autosampler-wandb-study5",
        storage="sqlite:///example.db",
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=10,  # Trials per process
        callbacks=[wandbc_callback] # Use the passed callback
    )
    best_trial = study.best_trial
    print(f"Process finished. Best trial value: {best_trial.value}")
    print(f"Process finished. Best trial params: {best_trial.params}")


if __name__ == '__main__':
    num_processes = 16

    # Initialize WeightsAndBiasesCallback *ONCE* in the main process
    wandb_kwargs = {"project": "optuna-autosampler-wandb-integration5"}
    wandbc = WeightsAndBiasesCallback(metric_name='score', wandb_kwargs=wandb_kwargs, as_multirun=False) # Ensure as_multirun=False
    wandb.define_metric("score", step_metric="score") # define metric here, use wandb.define_metric, not wandb.run.define_metric

    processes = []
    for _ in range(num_processes):
        # Pass the same callback instance to each process
        process = multiprocessing.Process(target=run_optimization_process, args=(wandbc,))
        processes.append(process)
        process.start() # Start the process

    for process in processes:
        process.join() # Wait for each process to finish

    print("All optimization processes finished.")