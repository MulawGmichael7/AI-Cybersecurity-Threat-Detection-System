"""
main.py
-------
Runs the full pipeline: generate data (if missing) -> train -> predict demo.

Run: python3 main.py
"""

import os
import sys
sys.path.append("src")

def main():
    """
    Run the complete cyber-threat detection pipeline.

    Steps:
    1. Create the dataset if it does not already exist.
    2. Train the machine learning model and print evaluation metrics.
    3. Run the prediction demo using a sample network flow.
    """
    if not os.path.exists("data/network_traffic.csv"):
        print("Dataset not found — generating it now...")
        from data.generate_data import generate
        generate().to_csv("data/network_traffic.csv", index=False)

    print("\n=== Training model ===")
    from src.train import train_and_evaluate
    train_and_evaluate()

    print("\n=== Running prediction demo ===")
    import runpy
    runpy.run_path("src/predict.py", run_name="__main__")

if __name__ == "__main__":
    main()
