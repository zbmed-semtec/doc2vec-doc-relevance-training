import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import utilities as utilities
import precision as precision


def run(best_params, args):

    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Train Data")

    # 2) Train the model with 80% of the data and best parameters
    start = time.time()
    model = utilities.createDoc2VecModel(train_pmids, train_docs, best_params)
    print(f"Time taken to train the model: {time.time() - start} seconds")
    print("RELISH Doc2Vec Model Generated.")
    print(model, "Model is being used.")

    # 3) Load the validation data from npy file
    val_pmids, val_docs = utilities.process_data_from_npy(args.valid)
    print(f"Retrieved RELISH Cleaned Validation Data")

    # 4) Generate the embeddings for validation dataset: pd.DataFrame for loaded docs
    val_embeddings_df = utilities.generate_embeddings(model, val_pmids, val_docs)
    print(f"RELISH Validation Embeddings Pickle File Generated.")

    # 5) Generate the cosine similarity validation matrix: pd.DataFrame for the generated embeddings
    val_similarity_df = utilities.get_similarity_scores(args.valid_ground_truth, val_embeddings_file)
    print(f"RELISH Validation Cosine Similarity Matrix Generated.")

    return val_similarity_df, val_embeddings_df, model