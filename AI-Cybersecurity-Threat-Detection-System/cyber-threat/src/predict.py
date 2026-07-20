"""
predict.py
----------
Loads the trained model and classifies a single network flow as
normal or attack. Run after train.py has produced src/model.joblib.
"""

import pandas as pd
import joblib


def predict_sample(sample: dict, model_path="src/model.joblib"):
    model = joblib.load(model_path)
    df = pd.DataFrame([sample])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    label = "ATTACK" if pred == 1 else "NORMAL"
    confidence = proba[pred]
    return label, confidence


if __name__ == "__main__":
    # Real example pulled from the dataset (not hand-crafted) — a flow the
    # trained model classifies as an attack with high confidence: elevated
    # error rates, several failed logins, and compromised-host indicators.
    example_flow = {
        "duration": 132.77, "src_bytes": 21060.0, "dst_bytes": 12112.0,
        "count": 29.0, "srv_count": 43.0, "serror_rate": 0.7252,
        "rerror_rate": 0.6478, "same_srv_rate": 0.2292, "diff_srv_rate": 0.6619,
        "dst_host_count": 49.0, "dst_host_srv_count": 32.0,
        "dst_host_same_srv_rate": 0.6494, "dst_host_diff_srv_rate": 0.5882,
        "dst_host_serror_rate": 0.2520, "num_failed_logins": 5.0,
        "logged_in": 1, "num_compromised": 3.0, "root_shell": 0,
        "num_file_creations": 2.0, "wrong_fragment": 1.0,
    }

    label, confidence = predict_sample(example_flow)
    print(f"Prediction: {label}  (confidence: {confidence:.1%})")
