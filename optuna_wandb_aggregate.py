import optuna
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed

def log_trial(trial):
    # Each process creates its own WandB run
    run = wandb.init(
        project="moz3",
        name=f"trial_{trial.number}",
        reinit=True,  # ensures a new run is created
        job_type="trial"
    )
    # Update configuration with the trial's parameters
    run.config.update(trial.params)
    # Log the objective (you can add more metrics if needed)
    run.log({"objective": trial.value}, step=0)
    run.finish()
    return f"Trial {trial.number} logged."

def parallel_log_trials(merged_study, max_workers=8):
    trials = merged_study.trials
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all logging jobs concurrently
        future_to_trial = {executor.submit(log_trial, trial): trial for trial in trials}
        for future in as_completed(future_to_trial):
            result = future.result()
            results.append(result)
            print(result)
    return results

def main():
    # Load the merged study (which contains all trials from your parallel runs)
    merged_study = optuna.load_study(
        study_name="merged_demo51",
        storage="sqlite:///merged_demo51.db"
    )

    # Log each trial as a separate run in parallel.
    # Adjust max_workers as appropriate for your machine.
    parallel_log_trials(merged_study, max_workers=16)
    print("Finished logging all trials in parallel to WandB.")

if __name__ == "__main__":
    main()
