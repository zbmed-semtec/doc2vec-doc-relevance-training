import os
import argparse
from optunaTuning import run_optuna_optimization
from train import run
import precision
import calculate_gain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input (train) file")
    parser.add_argument("-v", "--valid", help="Path to validation data file")
    parser.add_argument("-t", "--test", help="Path to test data file")
    parser.add_argument("-gv", "--valid_ground_truth", help="Path to valid ground truth .tsv file")
    parser.add_argument("-gt", "--test_ground_truth", help="Path to valid ground truth .tsv file")
    args = parser.parse_args()

    best_params, best_trial = run_optuna_optimization(args, n_trials=100, n_jobs=2)

    print("Finished Optuna optimization")
    similarity_file = run(best_params, args)

    output_directory = "output_doc2vec"
    precision_file = os.path.join(output_directory, "precision.tsv")
    dcg_file = os.path.join(output_directory, "dcg.tsv")
    idcg_file = os.path.join(output_directory, "idcg.tsv")
    ndcg_file = os.path.join(output_directory, "ndcg.tsv")

    # Generate and save the precision matrix
    ref_pmids, data = precision.read_file(similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data)
    precision.write_to_tsv(ref_pmids, matrix, precision_file)
    print("Precision matrix saved")

    # Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("DCG, IDCG, and NDCG matrices saved")



