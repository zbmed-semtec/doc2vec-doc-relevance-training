# Doc2Vec-Doc-relevance
This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the Doc2Vec technique with its application to the RELISH dataset. 


## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [Generate Embeddings](#generate-embeddings)
        - [Create Tagged Documents](#create-tagged-documents)
        - [Train and Optimize Doc2Vec models ](#train-and-optimize-Doc2Vec-models)
          - [Parameters](#parameters)
    2. [Doc2Vec Model Training and Similarity Matrix Computation on Split Dataset](#Doc2Vec-Model-Training-and-Similarity-Matrix-Computation-on-Split-Dataset)
    3. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
4. [Code Implementation](#code-implementation)
5. [Getting Started](#getting-started)
    - [Hyperparameter-Optimized Training with Split Dataset](#hyperparameter-optimized-training-with-split-dataset)
6. [Tutorial](#tutorial)

## About

 Our approach involves employing the [doc2vec](https://arxiv.org/pdf/1405.4053v2.pdf) model, which extends the popular word2vec technique to capture document-level semantics. By encoding documents and their textual content into fixed-length vectors, doc2vec facilitates similarity calculations and enables meaningful comparisons between documents. This approach is harnessed to derive insightful doc-2-doc recommendations within the realm of biomedical research, specifically employing the RELISH dataset. In order to do so, we employ the [doc2vec model](https://radimrehurek.com/gensim/models/doc2vec.html) from the [Gensim](https://radimrehurek.com/gensim/index.html) library.

## Input Data

The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the **RELISH.npy file**, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, the text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words.

## Pipeline

This section outlines the progression from generating document embeddings to conducting hyperparameter optimization and ultimately evaluating the effectiveness of the approach.

### Generate Embeddings
The following section outlines the process of generating document-level embeddings for each PMID of the RELISH corpus.

#### Create Tagged Documents 
In this initial step, we create  `TaggedDocuments `, which associates each PMID with a corresponding list of words. Here, we combine the abstract and title of each document into a unified paragraph (or document). This unified text serves as the input for our Doc2Vec model, allowing it to capture the semantic meaning of the entire document.

#### Train and Optimize Doc2Vec models 
In the second phase, we create and train Doc2Vec models with customizable hyperparameters to comprehend the connections between documents and words in a high-dimensional vector space. We aim to optimize these hyperparameters to establish the most effective relationship between cosine similarity and document relevance.

To accomplish this we begin by splitting the dataset into a training set and a testing set. The training set is then used to train the Doc2Vec model, where we explore various hyperparameters to optimize its performance. This optimization process is crucial for enhancing the model's ability to capture meaningful relationships between cosine similarity and document relevance. For each set of hyperparameters, a Doc2Vec model is trained on the training split. 

Following this, we evaluate the model's performance on the testing set using Precision@5 as our evaluation metric.

##### Parameters

+ **dm:** {1,0} Refers to the training algorithm. If dm=1, distributed memory is used otherwise, a distributed bag of words is used.
+ **vector_size:** It represents the dimensions of the generated embeddings, with options of 200, 300, and 400 in our case.
+ **window:** Represents the maximum distance between the current and predicted word, with values of 5,6 and 7 in our case.
+ **epochs:** Refers to the number of iterations over the training dataset and is set at 15 in this context.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm and is configured at a minimum of 5.

### Doc2Vec Model Training and Similarity Matrix Computation on Split Dataset
Following hyperparameter optimization, the next step involves training the Doc2Vec model with the optimal parameters on the training dataset. Then, embeddings are generated for the test dataset using this trained model. Subsequently, cosine similarity is calculated for the test dataset embeddings, providing a measure of similarity between pairs of documents based on their learned representations. 

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).


### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).

## Code Implementation


The [`main.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py) serves as a comprehensive wrapper function, supporting the creation of tagged documents, model generation, training, embedding generation, cosine similarity matrix calculation, precision calculation and gain calculation in one pipeline. Individual functions for each task are provided in the other two code scripts:

+ [`optuna_tuning.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/optuna_tuning.py): The code utilizes Optuna for hyperparameter optimization of a logistic regression classifier trained on similarity scores from document embeddings. It suggests hyperparameters for Doc2Vec, trains models, evaluates accuracy, and selects the best trial. The optimization process iterates over several trials, updating progress with a progress bar.
+ [`train.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py): This script trains a Doc2Vec model using specified hyperparameters, saves the model if specified, generates embeddings for test data, computes cosine similarity scores, and saves them to a file. It logs progress to a file specified by log_file.
+ [`utilities.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/utilities.py): Creation of tagged documents from input tokens, creation and training of Doc2Vec models, generation of embeddings, calculate cosine similarity, generate similarity matrix.
+ [`precision.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/precision.py): This script reads a TSV file containing cosine similarity pairs, calculates precision scores at various values of n for each PMID, and writes the results along with average precision scores to a new TSV file.
+ [`calculate_gain.py`](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/calculate_gain.py): This script calculates normalized discounted cumulative gain (nDCG) scores for relevance assessment based on cosine similarity values, sorts data accordingly, and writes results including average nDCG scores to a TSV file. It utilizes the cosine similarity matrix provided and performs operations per PMID.

## Getting Started

To get started with this project, follow these steps:

## Phase II - Split Dataset Training

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/doc2vec-doc-relevance.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/doc2vec-doc-relevance.git
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.9 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```
or
```
source deactivate
```

### Step 3: Dataset
- Download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN)
- Use [Download_Data.sh](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/Download_Dataset.sh) script to download the Split Dataset in the below mentioned format.

```
chmod +777 Download_Dataset.sh
./Download_Data.sh
#or
sh ./Download_Data.sh
```
- The data in the below-specified format

![image](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/assets/62026329/a8e84f15-2595-4292-a44c-70687dd9aea3)

### Step 4: Optimization Pipeline

This pipeline aims to optimize hyperparameters for a Doc2Vec model using Optuna, train the model with the optimal parameters, and evaluate its performance using precision at N (Precision@N) and normalized discounted cumulative gain (NDCG) metrics.

#### Pipeline Steps:

- **Hyperparameter Optimization**: Utilizes Optuna to search for the best hyperparameters for the Doc2Vec model.
- **Model Training**: Trains the Doc2Vec model with the optimal hyperparameters using 80% of the training split data.
- **Embedding Generation**: Generates embeddings for the remaining 20% of the test split data using the trained model.
- **Cosine Similarity Computation**: Calculates cosine similarities for the generated embeddings.
- **Precision@N Calculation**: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
- **NDCG Score Calculation**: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this [script](/code/train_model/main.py), and run the following command:

```
python3 code/train_model/main.py [-i INPUT] [-v VALIDATION_FILE] [-t TEST_FILE] [-gv VALIDATION_GROUND_TRUTH] [-gt TEST_GROUND_TRUTH] [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
```

You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
+ -v/ --valid :  File path to the RELISH Validation split dataset (.npy file format).
+ -t/ --test :  File path to the RELISH Test split dataset (.npy file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -c/  --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3)
+ -win/ --windows : 1- if using Windows systems; 0- if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

```
python3 code/train_model/main.py -i data/Split_Dataset/Data/train.npy -v data/Split_Dataset/Data/valid.npy -t data/Split_Dataset/Data/test.npy -gv data/Split_Dataset/Groundtruth/valid.tsv -gt data/Split_Dataset/Groundtruth/test.tsv -c 2 -win 0
```

Precision@N and NDCG scores are saved to TSV files in the following folder path: \output_2 (2 classes) and \output_3 (3 classes) for further analysis and reporting.

## Tutorial
A [tutorial](./docs/embeddings/) is accessible in the form of a Jupyter notebook for the generation of embeddings.

