import numpy as np
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_data(n_samples=1000):
    X = np.random.rand(n_samples, 5)
    y = (2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] +
         0.5 * X[:, 3] - 1.5 * X[:, 4] +
         np.random.normal(0, 0.1, n_samples))
    return X, y

def train_random_forest(trial):
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    # Ensure you are logged into WandB (either via CLI using wandb login or by setting your API key)
    wandb.login()
    
    # Setup WandB callback with your project configuration
    wandb_kwargs = {"project": "wssw"}
    wandb_callback = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    
    # Create a callback to stop after 100 completed trials across all instances
    terminate_callback = optuna.study.MaxTrialsCallback(n_trials=500, 
                                                        states=(optuna.trial.TrialState.COMPLETE,))
    
    # Use a shared SQLite backend to allow multiple processes to work on the same study.
    study = optuna.create_study(storage="sqlite:///wssw.db",
                                study_name="wssw",
                                load_if_exists=True,
                                direction="minimize")
    
    # Run optimization: multiple instances can be launched concurrently.
    study.optimize(train_random_forest, n_trials=800,
                   callbacks=[terminate_callback, wandb_callback])
    
    best_trial = study.best_trial
    print("Best Params:", best_trial.params)
    print("Best MSE:", best_trial.value)
