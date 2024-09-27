import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import utilities as utilities
import precision as precision
from sklearn.model_selection import KFold

def kfold_train(args, params_dict, n_splits):

    # Load the training data once
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)

    train_pmids = np.array(train_pmids, dtype=object)
    train_docs = np.array(train_docs, dtype=object)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # List to store precision scores for each fold
    precision_scores = []

    # ground_truth = utilities.process_ground_truth_from_npy(args.input)

    for fold_num, (train_index, val_index) in enumerate(kf.split(train_docs), start=1):
        # Split data into train and validation sets
        train_docs, valid_docs = train_docs[train_index], train_docs[val_index]
        train_pmids, valid_pmids = train_pmids[train_index], train_pmids[val_index]
        
        column_names = ["PMID1", "PMID2", "Value"]
        train_ground_truth = pd.read_csv(args.train_ground_truth, sep="\t", names = column_names, skiprows=1)

        valid_ground_truth = []

        for row in train_ground_truth.iterrows():
            pmid1 = row[0]
            pmid2 = row[1]

            if pmid1 in valid_pmids and pmid2 in valid_pmids:
                valid_ground_truth.append(row)

        valid_ground_truth = pd.DataFrame(valid_ground_truth)  

        input_data = {
            'train_docs': train_docs,
            'valid_docs': valid_docs,
            'train_pmids': train_pmids,
            'valid_pmids': valid_pmids,
            'kfold': fold_num
        }

        input_data['valid_ground_truth'] = valid_ground_truth

        # Train the model
        similarity_df, embeddings_df, model = run(params_dict, input_data)

        # Compute precision@5 for the validation set
        ref_pmids = similarity_df["PMID1"].unique()
        vector = precision.generate_vector(ref_pmids, similarity_df, args.classes)
        precision_5 = np.mean(vector, axis=0)[0]  # Take the precision@5 score

        precision_scores.append(precision_5)

    # Average the precision scores across all folds
    avg_precision_5 = np.mean(precision_scores)

    return avg_precision_5

def run(best_params, **input_data):

    # 1) Unpacking the training data and validation data
    train_pmids = input_data['train_pmids']
    train_docs = input_data['train_docs']
    valid_pmids = input_data['valid_pmids']
    valid_docs = input_data['valid_docs']
    valid_ground_truth = input_data['valid_ground_truth']
    Kfold = input_data['kfold']
    logging.info("Retrieved RELISH Cleaned Data")

    # 2) Train the model with 80% of the data and best parameters
    start = time.time()
    model = utilities.createDoc2VecModel(train_pmids, train_docs, best_params)
    logging.info(f"Time taken to train the model: {time.time() - start} seconds")
    logging.info("RELISH Hybrid Dord2Vec Model Generated.")
    logging.info("Model is being used.")

    # 3) Generate the embeddings: pd.DataFrame for validation data
    valid_embeddings_df = utilities.generate_embeddings(model, valid_pmids, valid_docs)
    logging.info(f"RELISH Validation Embeddings Pickle File Generated.")

    # 6) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(valid_ground_truth, valid_embeddings_df)
    logging.info(f"RELISH Validation Cosine Similarity Matrix Generated.")

    return similarity_df, embeddings_df, model