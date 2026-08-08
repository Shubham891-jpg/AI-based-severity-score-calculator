FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run pre-download to download and build cache for the ML model and NLTK data during image build
RUN python download_model.py

# Expose port (standard container port 8080)
EXPOSE 8080

# Run the startup script (which loads PORT from environment variable)
CMD ["python", "run_server.py"]
