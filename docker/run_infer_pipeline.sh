#!/bin/bash
set -e

echo "[INFO] Starting INFERENCE pipeline..."
python src/data_prep.py --input_dir /data --output_dir /data --inference_mode
python src/inference.py --input_dir /data --model_dir /models --output_dir /output
echo "[INFO] Inference pipeline completed successfully."
