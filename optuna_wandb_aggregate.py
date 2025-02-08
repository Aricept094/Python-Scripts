import optuna
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

def log_trial(trial):
    # Each process creates its own WandB run in offline mode;
    # this prevents hitting rate limits from simultaneous online connections.
    run = wandb.init(
        project="wsswo",
        name=f"trial_{trial.number}",
        reinit=True,       # ensures a new run is created
        job_type="trial",
        mode="offline"     # log offline to avoid network calls
    )
    # Log trial parameters and metrics
    run.config.update(trial.params)
    run.log({
        "objective": trial.value,
        "metric": trial.value,  # explicitly log metric
        "n_estimators": trial.params.get("n_estimators")  # explicitly log n_estimators
    }, step=0)
    run.finish()
    return f"Trial {trial.number} logged offline."

def parallel_log_trials(merged_study, max_workers=16):
    trials = merged_study.trials
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all logging jobs concurrently in parallel processes.
        future_to_trial = {executor.submit(log_trial, trial): trial for trial in trials}
        for future in as_completed(future_to_trial):
            result = future.result()
            results.append(result)
            print(result)
    return results

def sync_offline_runs(sync_path="./wandb"):
    # Sync all offline-run directories to the cloud.
    # Adjust the sync_path if your offline runs are stored in a different folder.
    result = subprocess.run(["wandb", "sync", sync_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error syncing offline runs:\n", result.stderr)
    else:
        print("Offline runs synced successfully:\n", result.stdout)

def main():
    # Load the merged study (which contains all trials from your parallel runs)
    merged_study = optuna.load_study(
        study_name="wssw",
        storage="sqlite:///wssw.db"
    )
    # Log each trial as a separate run offline in parallel.
    parallel_log_trials(merged_study, max_workers=16)
    print("Finished logging all trials offline to WandB.")
    
    # Once all trials are logged, sync them together to WandB.
    sync_offline_runs()  # Adjust the sync_path if needed

if __name__ == "__main__":
    main()
