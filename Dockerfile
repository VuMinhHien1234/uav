FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    torch torchvision \
    boto3 Pillow

WORKDIR /app

# Copy only what the training job needs
COPY config/ config/
COPY training/ training/

ENV PYTHONPATH=/app

CMD ["python3", "-m", "training.job"]
