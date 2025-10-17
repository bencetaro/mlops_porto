import pandas as pd
import argparse
import os
import joblib
import logging
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main(model_dir, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Loading model and datasets...")
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    test = pd.read_csv(os.path.join(input_dir, "test_preprocessed.csv"))
    test_ids = pd.read_csv(os.path.join(input_dir, "test_ids.csv"))

    logging.info("Generating predictions on test set...")
    test_pred = model.predict_proba(test)[:, 1]

    submission = pd.concat([test_ids, pd.DataFrame(test_pred, columns=["target"])], axis=1)
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    logging.info(f"Submission saved to {output_dir}/submission.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.model_dir, args.input_dir, args.output_dir)
