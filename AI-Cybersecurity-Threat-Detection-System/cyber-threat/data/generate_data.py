"""
generate_data.py
-----------------
Generates a synthetic network-traffic dataset for intrusion detection,
shaped like classic IDS datasets (NSL-KDD / CICIDS-style feature names).

Why synthetic? No internet access is available in this environment to
download NSL-KDD/CICIDS2017 directly. The statistical structure here is
built with scikit-learn's make_classification (controlled, reproducible,
genuinely learnable signal) and then mapped onto realistic network-flow
feature names so the project reads like a real IDS dataset.

To use a real public dataset instead (recommended for a stronger
portfolio piece later): download NSL-KDD or CICIDS2017 and replace
`data/network_traffic.csv`, keeping a `label` column (0 = normal,
1 = attack). Nothing else needs to change.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42

FEATURE_NAMES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_serror_rate", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "num_file_creations",
    "wrong_fragment",
]


def generate(n_samples=6000):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=14,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4],   # 60% normal, 40% attack traffic — realistic-ish imbalance
        flip_y=0.03,          # 3% label noise so it isn't trivially separable
        class_sep=1.7,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # Rescale features into plausible network-flow ranges instead of raw
    # standardized values from make_classification.
    df["duration"] = np.abs(df["duration"] * 50).round(2)
    df["src_bytes"] = np.abs(df["src_bytes"] * 5000).round(0)
    df["dst_bytes"] = np.abs(df["dst_bytes"] * 3000).round(0)
    df["count"] = np.abs(df["count"] * 20).round(0)
    df["srv_count"] = np.abs(df["srv_count"] * 15).round(0)
    for rate_col in ["serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
                      "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate"]:
        df[rate_col] = (df[rate_col] - df[rate_col].min()) / (df[rate_col].max() - df[rate_col].min())
    df["dst_host_count"] = np.abs(df["dst_host_count"] * 30).round(0)
    df["dst_host_srv_count"] = np.abs(df["dst_host_srv_count"] * 20).round(0)
    df["num_failed_logins"] = np.abs(df["num_failed_logins"]).round(0).clip(0, 5)
    df["logged_in"] = (df["logged_in"] > df["logged_in"].median()).astype(int)
    df["num_compromised"] = np.abs(df["num_compromised"]).round(0).clip(0, 3)
    df["root_shell"] = (df["root_shell"] > df["root_shell"].quantile(0.85)).astype(int)
    df["num_file_creations"] = np.abs(df["num_file_creations"]).round(0).clip(0, 4)
    df["wrong_fragment"] = np.abs(df["wrong_fragment"]).round(0).clip(0, 3)

    df["label"] = y  # 0 = normal, 1 = attack
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/network_traffic.csv", index=False)
    print(f"Generated data/network_traffic.csv with {len(df)} rows")
    print(df["label"].value_counts(normalize=True).rename("proportion"))
