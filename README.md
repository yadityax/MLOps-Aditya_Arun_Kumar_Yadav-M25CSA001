# Assignment-3: End-to-End Hugging Face Model Training & Docker



## Overview
This repository contains the complete pipeline for fine-tuning, evaluating, and deploying `distilbert-base-cased` model to classify Goodreads book reviews into 8 distinct genres. The project encompasses model training, local testing, and production evaluation, all executed entirely within fully containerized Docker environments.

## Steps Taken
Building this pipeline involved the following phases of development and deployment:

1. **Environment Setup & Dependency Management:**
   * Configured the `requirements.txt` file to ensure a consistent Python environment.
   * Installed core machine learning and NLP libraries, including `transformers`, `torch`, and `scikit-learn`.
2. **Model Training (DistilBERT):**
   * Leveraged the Hugging Face `Trainer` API to fine-tune the `distilbert-base-cased` model on the categorized Goodreads dataset, executed entirely within a local Docker container.
   * Configured training arguments to utilize GPU acceleration, seamlessly passed through to the Docker environment.
3. **Local Evaluation:**
   * Generated predictions on the test dataset from within the training container.
   * Calculated comprehensive performance metrics using Scikit-learn to output a detailed classification report.
4. **Model Deployment (Hugging Face Hub):**
   * Authenticated with the Hugging Face Hub using access tokens.
   * Pushed the fine-tuned model weights and the customized tokenizer to a public remote repository directly from the Docker environment.
   * Hugging Face Link - https://huggingface.co/m25csa001/distilbert-reviews-genres
5. **Dockerization for Production:**
   * Created a base `Dockerfile` to containerize the interactive training and local evaluation environment.
   * Built a separate production `Dockerfile.eval` to automate the process of pulling the live model from the Hugging Face Hub and running the evaluation suite without manual intervention.

## Model Performance 



The project generated two performance reports during the testing and deployment phases.

### 1. Classification Report
Detailed performance metrics across all 8 target genres:

| Genre | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| children | 0.66 | 0.70 | 0.68 | 200 |
| comics_graphic | 0.82 | 0.84 | 0.83 | 200 |
| fantasy_paranormal | 0.42 | 0.43 | 0.43 | 200 |
| history_biography | 0.58 | 0.59 | 0.58 | 200 |
| mystery_thriller_crime | 0.52 | 0.56 | 0.53 | 200 |
| poetry | 0.80 | 0.80 | 0.80 | 200 |
| romance | 0.63 | 0.56 | 0.59 | 200 |
| young_adult | 0.38 | 0.34 | 0.36 | 200 |
| **accuracy** | | | **0.60** | **1600** |
| **macro avg** | 0.60 | 0.60 | 0.60 | 1600 |
| **weighted avg** | 0.60 | 0.60 | 0.60 | 1600 |

### 2. Evaluation Result (Hub/Docker Evaluation)
Final performance metrics after containerization and deployment verification:

| Genre | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| children | 0.63 | 0.64 | 0.63 | 200 |
| comics_graphic | 0.89 | 0.82 | 0.86 | 200 |
| fantasy_paranormal | 0.45 | 0.53 | 0.49 | 200 |
| history_biography | 0.64 | 0.61 | 0.62 | 200 |
| mystery_thriller_crime | 0.58 | 0.56 | 0.57 | 200 |
| poetry | 0.75 | 0.80 | 0.77 | 200 |
| romance | 0.68 | 0.61 | 0.64 | 200 |
| young_adult | 0.40 | 0.41 | 0.40 | 200 |
| **accuracy** | | | **0.62** | **1600** |
| **macro avg** | 0.63 | 0.62 | 0.62 | 1600 |
| **weighted avg** | 0.63 | 0.62 | 0.62 | 1600 |

## How to Run

**1. Interactive Training Environment**
Build and run the primary Docker container to execute the training scripts locally.
```bash
docker build -t mlops-exp3 .
docker run --rm -it mlops-exp3 python train.py
docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -v $(pwd):/MLOPS_EXP \
  mlops-exp3
