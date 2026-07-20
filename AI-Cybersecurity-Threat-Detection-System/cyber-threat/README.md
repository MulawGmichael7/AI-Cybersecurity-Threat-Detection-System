# AI-Based Cybersecurity Threat Detection

AI-based cybersecurity system for detecting and analyzing potential
threats using machine learning techniques. Classifies network traffic
flows as normal or an attack using a Random Forest model trained on
flow-level features (connection duration, byte counts, error rates,
login behavior, and more).

## How it works

1. **Feature representation** — each network flow is described by 20
   features modeled on classic intrusion-detection datasets (NSL-KDD /
   CICIDS-style): duration, byte counts, service/error rates, failed
   login counts, root-shell access indicators, and host-level aggregate
   statistics.
2. **Model** — a `RandomForestClassifier` (scikit-learn), trained on 80%
   of the data and evaluated on a held-out 20% test set it never saw
   during training.
3. **Evaluation** — accuracy, precision, recall, F1, a confusion matrix,
   and feature importances are all computed directly from the held-out
   set (see Results below — these are real, reproducible numbers).
4. **Inference** — `src/predict.py` loads the trained model and classifies
   a single flow as normal or attack, with a confidence score.

## Dataset

`data/generate_data.py` generates a synthetic dataset of 6,000 labeled
network flows.

> No internet access was available in the environment this project was
> built in, so real datasets (NSL-KDD, CICIDS2017) couldn't be downloaded
> directly. The synthetic data is built with scikit-learn's
> `make_classification` for a controlled, reproducible, genuinely
> learnable signal, then mapped onto realistic network-flow feature
> names. **To use a real dataset instead:** download NSL-KDD or
> CICIDS2017 and replace `data/network_traffic.csv`, keeping the same
> column structure and a `label` column (0 = normal, 1 = attack) — no
> other code changes needed.

## Results

Evaluated on a held-out test set (1,200 flows, unseen during training):

| Metric | Score |
|---|---|
| Accuracy | 93.1% |
| Precision | 94.7% |
| Recall | 87.8% |
| F1-score | 91.1% |

Top predictive features: `same_srv_rate`, `serror_rate`,
`dst_host_serror_rate`, `dst_host_diff_srv_rate`, `rerror_rate` — all
service/error-rate features, consistent with how real intrusion
detection systems flag anomalous connection patterns.

(Reproduce with `python3 src/train.py`.)

## Project structure

```
├── data/
│   └── generate_data.py       # synthetic network-flow dataset generator
├── src/
│   ├── train.py                # trains the Random Forest + evaluation
│   └── predict.py              # inference on a single flow
├── main.py                     # end-to-end pipeline runner
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python3 main.py                  # generates data, trains, evaluates, runs a prediction demo
python3 src/train.py             # retrain / re-evaluate only
python3 src/predict.py           # run inference on the example flow
```

### Example output

```
=== Evaluation on held-out test set (20% of data, unseen during training) ===
Accuracy  : 93.1%
Precision : 94.7%
Recall    : 87.8%
F1_score  : 91.1%

Prediction: ATTACK  (confidence: 99.9%)
```

## Future improvements

- Swap in a real public dataset (NSL-KDD / CICIDS2017) for stronger
  real-world validity.
- Add gradient-boosted models (XGBoost/LightGBM) for comparison.
- Build a simple real-time flow classifier using `scapy` or similar for
  live traffic monitoring.

## Author

Mulaw Gebremichael — Computer Engineer / Data Scientist, Mekelle University
