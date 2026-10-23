# Homework 5 — Deployment

Homework for the model deployment module of the DataTalksClub Machine Learning Zoomcamp 2026.

## Topics

- Python environment and dependency management with uv
- Model serialization and loading with Pickle
- Making predictions with a pretrained ML pipeline
- Building REST APIs with FastAPI
- Serving models with Uvicorn
- Sending HTTP requests for predictions
- Creating Docker images
- Running containerized ML applications

## Model

`pipeline.bin`

The homework uses a pretrained machine learning pipeline to predict whether a potential customer will convert based on their lead characteristics.

The focus is on loading and serving an existing model through an API and deploying it using Docker.

## Running

From the repository root:

### Local

```bash
uv run python hw5/predict.py
```

### Docker

Build the image:

```bash
docker build -t zoomcamp-hw5 -f hw5/Dockerfile .
```

Run the container:

```bash
docker run --rm -p 9696:9696 zoomcamp-hw5
```

The prediction API is available at `http://localhost:9696`.
