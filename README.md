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

# for contributing
pip install -r dev-requirements.txt
chmod +x run_code_compliance.sh
```

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