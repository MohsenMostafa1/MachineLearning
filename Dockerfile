# Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data ./data
COPY scripts ./scripts

CMD ["python", "scripts/train_tabular.py"]  # Run Step 1 by default
