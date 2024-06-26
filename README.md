# Token Classification from French census archives between 1836 and 1936

This project aims to structure and classify information on individuals contained in a corpus of documents. The goal is to implement a system capable of accurately identifying and categorizing data such as surname, first name, age, year of birth, relationship to head of household, nationality, and many others. This classification process would not only allow data to be organized more efficiently, but also enable more in-depth and targeted analyses.

Here is the structure of this repository:
- `data/`: This directory should contain the datasets and data-related resources. First, create a `data` folder and download the files _tokens.yml_ and _entities.json_ on the project website. Then, download the *first_name_with_sex.csv* and _georef-france-communes.csv_ files to work on the external data part (https://public.opendatasoft.com/explore/dataset/georef-france-commune).
- `src/`: Source code for the project.
  - `load_data.py`: A Python script for loading data.
  - `preprocessing.py`: A Python script dedicated to data preprocessing steps.
  - `experiment_helper.py`: A Python script that contain helper functions for running experiments.
  - `stat_des.ipynb`: A Jupyter notebook used for statistical descriptions, contains exploratory data analysis.
  - `experiment.ipynb`: A Jupyter notebook used for finetuning NLP models.
- `requirements.txt`: A text file containing a list of Python packages required to run the project. 
