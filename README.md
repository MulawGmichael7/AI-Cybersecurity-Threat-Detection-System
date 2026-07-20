# AI-Based Cybersecurity Threat Detection System

## Overview
This repository contains an AI-powered cybersecurity threat detection project that uses machine learning to classify network traffic as either normal or malicious. The system is built around a Random Forest classifier and a synthetic intrusion-detection dataset that mirrors the structure of real network-flow security datasets.

## Project Goal
The main goal of this project is to:
- detect anomalous or suspicious network behavior,
- classify network flows as normal or attack,
- provide a simple, reproducible machine learning workflow for cybersecurity analysis.

## How the project works
The pipeline is organized in a simple end-to-end flow:

1. Data generation
   - The function `generate()` in `data/generate_data.py` creates a synthetic dataset of labeled network flows.
   - It simulates realistic IDS-style features such as connection duration, byte counts, error rates, login failures, and host-level behavior.

2. Training
   - The function `load_data()` in `src/train.py` reads the dataset and separates it into features (`X`) and labels (`y`).
   - The function `train_and_evaluate()` trains a `RandomForestClassifier`, evaluates it on a held-out test set, prints important metrics, and saves the trained model as `src/model.joblib`.

3. Prediction
   - The function `predict_sample()` in `src/predict.py` loads the saved model and uses it to classify a single network-flow sample.
   - It returns both the predicted label and the model’s confidence score.

4. End-to-end execution
   - The `main()` function in `main.py` acts as the project entry point. It checks whether the dataset exists, generates it if necessary, trains the model, and runs the prediction demo.

## Function explanations

### `main()`
Runs the complete pipeline:
- checks for the presence of the dataset,
- creates the dataset if it is missing,
- trains the model,
- executes the prediction example.

### `load_data()`
Reads the dataset from CSV and splits it into:
- `X`: all input features,
- `y`: the target label column.

### `train_and_evaluate()`
Trains the Random Forest model and evaluates its performance by printing:
- accuracy,
- precision,
- recall,
- F1-score,
- confusion matrix,
- classification report,
- top feature importances.

It also saves the trained model for later use.

### `predict_sample()`
Loads the saved model and predicts whether a single input sample represents:
- `NORMAL` traffic, or
- `ATTACK` traffic.

It also returns the confidence score associated with the decision.

### `generate()`
Generates synthetic network-flow data with realistic feature ranges and a binary label column.
This makes the project runnable even when public cybersecurity datasets are unavailable.

## Dataset
The project uses a synthetic dataset generated from `data/generate_data.py`.
The generated file is stored as:
- `data/network_traffic.csv`

The dataset contains flow-like features such as:
- duration,
- source and destination bytes,
- connection counts,
- error and service rates,
- failed login counts,
- compromised host indicators.

## Project structure

```text
AI-Cybersecurity-Threat-Detection-System/
├── README.md
├── cyber-threat/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   ├── data/
│   │   └── generate_data.py
│   └── src/
│       ├── train.py
│       └── predict.py
└── .gitignore
```

## Running the project

```bash
pip install -r requirements.txt
python3 main.py
```

### What `python3 main.py` does
- creates the dataset if needed,
- trains the model,
- evaluates the model on a test split,
- runs the prediction demo.

## Example results
The project reports key classification metrics such as accuracy, precision, recall, and F1-score on a held-out test set. These values are reproducible and are useful for comparing model performance over time.

## Future improvements
- Replace the synthetic dataset with a real public IDS dataset such as NSL-KDD or CICIDS2017.
- Add more advanced models such as XGBoost or LightGBM.
- Extend the system to real-time anomaly monitoring.
- Add a web dashboard or API for live threat classification.

## Author
Mulaw Gebremichael
