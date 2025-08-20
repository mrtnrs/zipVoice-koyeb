# Use NVIDIA CUDA base image with Ubuntu and Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    ca-certificates \
    libsndfile1 \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set workdir
WORKDIR /app

# Copy app code
COPY . /app

# Pre-download Hugging Face ZipVoice model to avoid re-download at runtime
# Ensure dependencies are installed
RUN pip install --no-cache-dir huggingface_hub

# Pre-download Hugging Face ZipVoice model
RUN python3 -m huggingface_hub download csukuangfj/zipvoice zipvoice_distill.pt --local-dir /models/zipvoice


# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
