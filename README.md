# Computational Intelligence Lab 2024
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
Project work for the ETHZ CIL Collaborative Filtering 2024 competition held on kaggle. The goal of the project is to correctly predict the movie ratings for given (user, movie)-pairs.

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
- ```data/sampleSubmission.csv``` is a sample submission file. It holds the Ids, for which the predictions have to be made.

## ▶️Run Experiments

Here, you can find all scripts used to reproduce the results of our different methods. Depending on your system, you might need to replace forward slashes "/" with backward slashes "\".

### Neural Collaborative Filtering (MLP-based)
```python cil_project\neural_filtering\training\ncf_procedure.py --embedding 128 --hidden 384 --batch_size 512```

## 📊 Results
<div style="text-align: center;">

| Method   | Local RMSE (CV) | Public RMSE |
|:---------|:---------------:|:-----------:|
| Method 1 |      0.85       |    0.82     |
| Method 2 |      0.88       |    0.85     |
| Method 3 |      0.90       |    0.87     |

</div>