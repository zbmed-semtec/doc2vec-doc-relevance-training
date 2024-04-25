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
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```
### Step 3: Dataset
- Download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN)
- Use [Download_Data.sh](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/Download_Dataset.sh) script to download the Split Dataset in the below mentioned format.

```
chmod +777 Download_Data.sh
./Download_Data.sh
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
python3 code/train_model/main.py -i data/Split_Dataset/Data/train.npy -v data/Split_Dataset/Data/valid.npy -t data/Split_Dataset/Data/test.npy -gv data/Split_Dataset/Ground_truth/relish_ground_truth_valid.tsv -gt data/Split_Dataset/Ground_truth/relish_ground_truth_test.tsv -c 2 -win 0
```

Precision@N and NDCG scores are saved to TSV files in the following folder path: \output_2 (2 classes) and \output_3 (3 classes) for further analysis and reporting.
