# Computational Intelligence Lab 2024 (Team Winterhold)
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
Project work for the ETH Zurich CIL Collaborative Filtering 2024 competition held on kaggle. 
For a total of 10'000 users and 1'000 movies, the dataset consists of 1'176'952 ratings ranging from 1 to 5.
Our goal is to predict the ratings for unobserved (user, movie)-pairs.
For this purpose, we implement an ensemble approach combining predictions from various models including baseline models such as ALS, SVT, SVP, and novel ideas in neural collaborative filtering and Bayesian factorization machines.
Using stacking, we achieve a public score on a hidden test set of 0.9649. Find the report [here](report.pdf).

## ⚡️ Quickstart
For local development, you need to have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. Then, run
```bash
conda env create -f ./environment.yml
conda activate cil-project

# install the project
pip install -e .

# for contributing
pip install -r dev-requirements.txt
chmod +x run_code_compliance.sh
```

If the environment.yml file is updated, additionally run
```
conda env update -f environment.yml --prune
pip install -e .
```
to keep the dependencies up to date.

## 🤝 Contributing
In order to keep the project organized, we propose the following workflow for contribution:
1. Create new issue.
1. Create new branch of the form ```<user>/#<issue_number>_<issue_title>```, e.g., ```maxmuster/#10_fix_SVM```.
1. Create a new PullRequest from this branch and link the issue.
1. Work on the PullRequest.
1. Upon completion, merge the changes with the main branch.

## 📂 Project Organization
- ```data/data_train.csv``` holds the training data with (Id,Prediction) tuples. The Id consists of a given (user, movie) rating. For example r44_c1 refers to the rating of the 44-th user for the 1st movie. The movie ratings are integer values ranging from 1-5.
- ```data/sampleSubmission.csv``` is a sample submission file. It holds the Ids of the hidden test set, for which the predictions have to be made.

## ▶️ Run Experiments

Here, you can find the scripts used to reproduce the results of our different methods. Depending on your system, you might need to replace forward slashes "/" with backward slashes "\\".

### Dataset Preparation
This allows the user to split the dataset such that it can be used for blending or stacking.
```bash
python cil_project/dataset_creator.py -n <name> --split <split> [--base <base dataset>] [--shuffle]
```
where:
- `-n <name>` is the name of the dataset to be created.
- `--split <split>` is the ratio of the dataset to be used for training (if floating point) or the number `k` of sets for k-fold cross-validation.
- `--base <base_dataset>` is the name of the dataset to be split. If not specified, the whole dataset is used.
- `--shuffle` shuffles the dataset before splitting.

### Baseline Models (ALS, SVP, SVT)
```bash
python cil_project/baseline_models/training/training_procedure.py [command line arguments]
```
where `[command line arguments]` can be the following:
- `--model <model_name>` - specifies the method to be used. The following options are available: `ALS`, `SVP`, `SVT`.
- `--blending <dataset> <val_dataset>` - prepares the model to be used for blending by training on the training dataset and predicting the validation and submission datasets.
- `--stacking <dataset_base_name> <val_dataset_base_name>` - prepares the model to be used for stacking by performing k-fold cross-validation and predicting the submission dataset.
- `--verbose` - prints additional information during training and testing.

### Neural Collaborative Filtering (NCF) & SVD++
These are all methods that involve training of neural networks.
Therefore, they share a common structure and can be run with the same command.
```bash
python cil_project/neural_filtering/training/<procedure> [command line arguments]
```
where `<procedure>` is the name of the procedure you want to run. The following procedures are available:
- `autoencoder_training_procedure.py` - the Autoencoder
- `ncf_gmf_training_procedure.py` - the GMF part of the NCFCombined model
- `ncf_mlp_training_procedure.py` - the MLP part of the NCFCombined model
- `ncf_combined_training_procedure.py` - the NCFCombined model
- `ncf_training_procedure.py` - the MLP-based NCF model
- `ncf_kan_training_procedure.py` - the KAN-based NCF model
- `svdpp_training_procedure.py` - the SVD++ model

Using `[command line arguments]`, one can specify, in which mode(s) the model should be run. The following options are available:
- `--test` - tests the model performance on a random split of the whole dataset (useful hyperparameter tuning).
- `--train [<dataset>] [<val_dataset>]` - trains the model on the given dataset (entire dataset if not specified) and validates on the validation dataset.
- `--checkpoint_granularity <number>` - specifies the number of epochs after which a checkpoint is saved (for training or testing).
- `--blending <dataset> <val_dataset` - prepares the model to be used for blending by training on the training dataset and predicting the validation and submission datasets.
- `--stacking <dataset_base_name> <val_dataset_base_name` - prepares the model to be used for stacking by performing k-fold cross-validation and predicting the submission dataset.

### BFM Models
BFM models can be trained using the following command:
```bash
python cil_project/bayesian_factorization_machines/bfm_training_procedure.py [command line arguments]
```
where the `[command line arguments]` involve:
- `--rank <number>` - specifies the rank of the BFM.
- `--iterations <number>` - the number of iterations for BFM training.
- `--kfold <k>` - the number used for k-fold cross-validation.
- `--grouped` - whether to use grouped features.
- `--implicit` - whether to use implicit features.
- `--statistics` - whether to use statistical features.
- `--op` - whether to use ordinal probit.
- `--kmeans` - whether to use Kmeans.
- `--num_clusters <number>` - the number of clusters for Kmeans.
- `--num_bins <number>` - the number of bins for statistical features.

Additionally, each BFM model can be run in blending or stacking mode. If none of them is specified, the model is trained on the provided dataset. The following commands can be used:
- `--blending <dataset> <val_dataset>` - prepares the model to be used for blending by training on the training dataset and predicting the validation and submission datasets.
- `--stacking <dataset_base_name> <val_dataset_base_name>` - prepares the model to be used for stacking by performing k-fold cross-validation and predicting the submission dataset.
- `--dataset <dataset>` - specifies the dataset to be used for training. If none specified, trains the model on the entire dataset.

In order to reproduce our results, consider the following models.

BFM (Base):
```bash
python cil_project/bayesian_factorization_machines/training/bfm_training_procedure.py --rank 15 --iterations 1000
```

BFM (Implicit):
```bash
python cil_project/bayesian_factorization_machines/training/bfm_training_procedure.py --rank 15 --iterations 1000 --grouped --implicit
```

BFM (Implicit + op):
```bash
python cil_project/bayesian_factorization_machines/training/bfm_training_procedure.py --rank 15 --iterations 1000 --grouped --implicit --op
```

BFM (Implicit + op + Kmeans):
```bash
python cil_project/bayesian_factorization_machines/training/bfm_training_procedure.py --rank 15 --iterations 1000 --grouped --implicit --op --kmeans --num_clusters 4
```

### Ensembling
The base command for performing an ensemble of methods is the following:
```bash
python cil_project/run_ensembler.py --models <model1> <model2> ... [--regressor <regressor>] [--val <val_name>]
```
Depending on the provided arguments, the script either performs:
- Blending: if a regressor is provided and one particular validation dataset is specified.
- Stacking: if a regressor is provided and the base name of the `k` validation datasets is given.
- Averaging: if no regressor is provided, a simple average of the models' predictions is performed.

## 📊 Results
<div align="center">

| Method                         | Local RMSE (CV) | Public RMSE |
|:-------------------------------|:---------------:|:-----------:|
| ALS                            |     0.9899      |   0.9876    |
| SVP                            |     0.9885      |   0.9860    |
| SVT                            |     0.9870      |   0.9844    |
| SVD++                          |     0.9784      |   0.9759    |
| Autoencoder                    |     0.9819      |   0.9799    |
| NCFCombined                    |     0.9826      |   0.9820    |
| _GMF (in NCFCombined)_         |     0.9856      |   0.9906    |
| _MLP (in NCFCombined)_         |     0.9878      |   0.9844    |
| MLP-based NCF                  |     0.9843      |   0.9806    |
| KAN-based NCF                  |     0.9834      |   0.9807    |
| BFM (Base)                     |     0.9776      |   0.9750    |
| BFM (Implicit)                 |     0.9701      |   0.9676    |
| BFM (Implicit + op)            |     0.9682      |   0.9661    |
| BFM (Implicit + op + Kmeans)   |     0.9680      |   0.9660    |
| Blending (Linear Regression)   |        -        |   0.9651    |
| Stacking (Linear Regression)   |        -        | **0.9649**  |

</div>