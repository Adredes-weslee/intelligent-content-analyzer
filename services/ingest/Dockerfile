FROM python:3.10-slim

# Install dependencies
COPY ../../requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy service code and shared modules
COPY ./app /app/app
COPY ../../shared /app/shared

WORKDIR /app
ENV PYTHONPATH=/app

# Expose port (not strictly necessary in Compose as it maps ports)
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]