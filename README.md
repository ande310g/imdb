# IMDb
This repository contains the code and experiments for the exam project “***Forecasting IMDb movie popularity: A supervised machine learning approach***”, developed for the *Applied Machine Learning (CINTO2401E)* course. The objective is to predict a movie’s post-release popularity on IMDb using only information available before release. It uses the IMDb's public dataset for non-commercial use train and evaluate multiple different models, with changing target variables. The two target variables that have been tried is: num_votes*avg_rating and num_votes.

Multiple supervised regression models are implemented and compared, including linear baselines (Linear, Ridge, Lasso, Polynomial) and tree-based ensemble models (Random Forest and XGBoost). The repository is organized into separate folders for each model and experimental variation to support reproducibility and systematic comparison.

Models are evaluated using MAE, RMSE, and R² on a chronological train/test split to reflect real-world prediction scenarios. Overall, the project aims to balance predictive performance with interpretability and treats the models as decision-support tools rather than definitive predictors of success.

# Prerequists
## Python version
For development version 3.14.0 was used as well as the following libraries:
| **Library**  | **Version** |
|--------------|-------------|
| NumPy        | 2.3.4       |
| Pandas       | 2.3.3       |
| Scikti-learn | 1.7.2       |
| SciPy        | 1.16.2      |
| Optuna       | 4.6.0       |
| Matplotlib   | 3.10.7      |
| JobLib       | 1.5.2       |
| tqdm         | 4.67.1      |
## Setup the data saving stages
The project is structured so that all models are built using a shared base dataset, while each model also maintains its own dedicated data folder for training and testing data.

Before running the project, the following directories must be created:
- ./src/common_datasets
- ./src/Lasso/data
- ./src/Linear_Regression/data
- ./src/Polynomial_regression/data
- ./src/Random_forrest/data
- ./src/XGBoost_First_Iteration/data
- ./src/XGBoost_Log_Transformed/data
- ./src/XGBoost_Number_Votes/data
- ./src/XGBoost_Over_1000_Votes/data
- ./src/XGBoost_Tweedie/data

## Download the data
Download the different .tsv files from [ImdB non-commerical datasets](https://developer.imdb.com/non-commercial-datasets/). Unzip them a put them into a folder called **data** at the root of the project. So your project should have a folder structure that looks like this ./data/*.tsv. 

# Run order
The order the scripts need to be runned at:
- First run "./src/Merging_TSV_files.ipynb"
- Second run the 3 following scripts: "./src/Preparing_dataset_original.ipynb", "./src/Preparing_dataset_log_scaled.ipynb" and "./src/Preparing_dataset_log_scaled_number_votes.ipynb"
- From here the different models can be runned independently by where each folders scripts is run in the following order "./src/"model"/Splitting_dataset.ipynb", "./src/"model"/Learning.ipynb" and then "./src/"model"/Testing.ipynb"
# Notebooks
Below are the different notebooks explained of what they do, the 3 model notebooks being "Learning.ipynb", "Splitting_dataset.ipny" and "Testing.ipynb" are explained here where the model variations are explained under the model.  
## Merging_TSV_files.ipynb
This notebook prepares a clean, the dataset from the official IMDb TSV exports. It incrementally loads the large raw files in chunks. The individual IMDb tables (titles, crew, ratings, principals, and names) are then merged using inner joins to ensure consistent and complete records across entities. The result is a CSV file that links movies with people, ratings, and metadata.

## Preparing_dataset_*.ipynb
These 3 notebooks do the approximately the same things but with little variations descriped beneath. These 3 notebooks takes the previously merged IMDb dataset and performs feature engineering to create a movie-level modeling table. They clean and convert year and runtime fields, and enforces a consistent temporal ordering to ensure that all historical features are based only on prior information.

The core of the script computes cumulative and top-3 prior performance metrics for both directors and actors, capturing historical success without leaking information from the current movie. Actor-level data is then reshaped from long to wide format so that each movie appears once, with a fixed number of top-billed actors represented as columns. Finally, additional time-based features (e.g. years since release) are added, and the resulting dataset is saved as a CSV for further modeling.

- **Preparing_dataset_original.ipynb**; this creates a dataset where the target variable is average rating multiplied with number reviews.
- **Preparing_dataset_log_scaled.ipynb**; to combat the issue that the data is heavily skewed towards the bottom, this script log transforms the target variable to create a better distribution.
- ***Preparing_dataset_log_scaled_number_votes.ipnyb**; this changes the target variable from being average rating multiplied with number of reviews, to instead being number of reviews. The target variable is still log transformed.

## Splitting_dataset.ipynb
