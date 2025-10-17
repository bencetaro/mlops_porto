#!/bin/bash
set -e

echo "[INFO] Starting TRAINING pipeline..."
python src/data_prep.py --input_dir /data --output_dir /data
python src/train_model.py --input_dir /data --output_dir /output
python src/evaluate.py --input_dir /data --model_dir /output --output_dir /output
echo "[INFO] Training pipeline completed successfully."
