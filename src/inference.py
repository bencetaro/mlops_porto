import os
import argparse
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main(input_dir, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting inference pipeline...")

    unseen_pth = os.path.join(input_dir, "unseen_preprocessed.csv")
    ids_pth = os.path.join(input_dir, "unseen_ids.csv")
    model_pth = os.path.join(model_dir, "model.pkl")

    assert os.path.exists(unseen_pth), f"Missing file: {unseen_pth}"
    assert os.path.exists(model_pth), f"Missing file: {model_pth}"

    logging.info("Loading preprocessed data and model...")
    data = pd.read_csv(unseen_pth)
    model = joblib.load(model_pth)

    logging.info(f"Data shape for inference: {data.shape}")
    start = datetime.now()
    preds = model.predict_proba(data)[:, 1]
    elapsed = datetime.now() - start
    logging.info(f"Inference done in {elapsed.total_seconds():.2f} seconds.")

    output_df = pd.DataFrame({"prediction": preds})
    if os.path.exists(ids_pth):
        ids = pd.read_csv(ids_pth)
        output_df = pd.concat([ids, output_df], axis=1)

    out_file = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(out_file, index=False)
    logging.info(f"Predictions saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.model_dir, args.output_dir)
