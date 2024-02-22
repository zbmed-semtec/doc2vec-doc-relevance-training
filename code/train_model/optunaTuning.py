import os
import optuna
import argparse
import logging
import pandas as pd
import numpy as np
from train import run
import utilities as utilities
from tqdm import tqdm
import precision as precision

log_file = "Optuna.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

model_directory = "Model"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

def objective_wrapper(args):
    def objective(trial):
        # Suggest hyperparameters for Doc2Vec
        vector_size = trial.suggest_int('vector_size', 200, 300)
        window = trial.suggest_int('window', 5, 15)
        min_count = trial.suggest_int('min_count', 1, 5)
        epochs = trial.suggest_int('epochs', 10, 50)
        workers = trial.suggest_int('workers', 2, 8)

        # Use args here as needed, e.g., args.input, args.test
        params = {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "workers": workers
        }
        # Assume run() trains the model and returns the path to a file with similarity scores
        model, similarity_file = run(params, args)
        
        ref_pmids, data = precision.read_file(similarity_file)
        matrix = precision.generate_matrix(ref_pmids, data)

        precision_5 = list(np.mean(matrix, axis=0).round(4))

        return precision_5
    return objective

def run_optuna_optimization(args, n_trials=3):
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
        study.optimize(objective_wrapper(args), n_trials=n_trials, callbacks=[callback])
    print('Best values:', study.best_trial.values)
    print('Best trial:', study.best_trial.params)
    logging.info('Best trial: %s', study.best_trial.params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input (train) file")
    parser.add_argument("-t", "--test", help="Path to test file")
    parser.add_argument("-gt", "--ground_truth", help="Path to ground truth .tsv file")
    args = parser.parse_args()
    
    run_optuna_optimization(args)
