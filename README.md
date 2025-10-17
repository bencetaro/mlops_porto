# End-to-end MLOps pipeline

This repo demonstrates an end-to-end Docker-based MLOps pipeline for the Porto Seguro Safe Driver dataset.

### Docker build & run:

#### Build image:
  docker build -f docker/Dockerfile -t porto-mlops .

#### Run containers with docker (automated bash setup):
For training (Data prep -> Train -> Eval)
  docker run -v $(pwd)/data:/data -v $(pwd)/output:/output porto-mlops:latest ./docker/run_train_pipeline.sh
For inference (Data prep -> Inference)
  docker run -v $(pwd)/data:/data -v $(pwd)/models:/models -v $(pwd)/output:/output porto-mlops:latest ./docker/run_infer_pipeline.sh
