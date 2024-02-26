import os
import time
import argparse
import logging
import utilities as utilities


log_file = "Doc2Vec_Split_data.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def run(best_params, args, save_model=False):
    # Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Data")
    logging.info("Retrieved RELISH Cleaned Data")

    start = time.time()
    # Train the model with 80% of the data and best parameters
    model = utilities.createDoc2VecModel(train_pmids, train_docs, best_params)
    end = time.time()
    print(f"Time taken to train the model: {end - start} seconds")
    logging.info("Time taken to train the model: {end - start} seconds")
    print("RELISH Doc2Vec Model Generated")
    logging.info("RELISH Doc2Vec Model Generated")

    print(model, "Model is being used.")

    if save_model:
        # Save the model
        utilities.saveDoc2VecModel(model, "output_doc2vec/best_model.model")


    # Load the test data
    test_pmids, test_docs = utilities.process_data_from_npy(args.test)
    print("Retrieved RELISH Cleaned Data")
    logging.info("Retrieved RELISH Cleaned Data")

    # Define a directory for storing embeddings
    embeddings_directory = "embeddings/embeddings_doc2vec"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)

    embeddings_file = os.path.join(embeddings_directory, "test_embeddings_pickle.pkl")

    # Generate the embeddings
    utilities.generate_embeddings(model, test_pmids, test_docs, embeddings_file)
    print("RELISH Embeddings Pickle File Saved")
    logging.info("RELISH Embeddings Pickle File Saved")

    # Define the directory for storing similarity results
    output_directory = "output_doc2vec"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate and save the cosine similarity matrix
    similarity_file = os.path.join(output_directory, "cosine_similarity.tsv")
    utilities.get_similarity_scores(args.ground_truth, embeddings_file, similarity_file)
    print("RELISH Cosine Similarity Matrix Saved")
    logging.info("RELISH Cosine Similarity Matrix Saved")

    return similarity_file