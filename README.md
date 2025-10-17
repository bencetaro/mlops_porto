# End-to-end MLOps pipeline

This repo demonstrates an end-to-end Docker-based MLOps pipeline for the Porto Seguro Safe Driver dataset.

### Docker build & run:

#### Build image:
    docker build -f docker/Dockerfile -t porto-mlops .

#### Run containers with docker (automated bash setup):
Training script (Data prep -> Train -> Eval)

    docker run -v $(pwd)/data:/data -v $(pwd)/output:/output porto-mlops:latest ./docker/run_train_pipeline.sh

Inference script (Data prep -> Inference)

    docker run -v $(pwd)/data:/data -v $(pwd)/models:/models -v $(pwd)/output:/output porto-mlops:latest ./docker/run_infer_pipeline.sh

Structure:

    safe_driver_mlops/
    ├── data/
    │   ├── train.csv.zip
    │   ├── test.csv.zip
    │   └── unseen.csv
    ├── src/
    │   ├── data_prep.py
    │   ├── train_model.py
    │   ├── evaluate.py
    │   └── inference.py
    ├── docker/
    │   ├── run_train_pipeline.sh
    │   ├── run_infer_pipeline.sh
    │   └── Dockerfile
    ├── requirements.txt
    └── README.md

