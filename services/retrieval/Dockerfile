FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY . /app


ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn services.retrieval.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]