# Use NVIDIA CUDA base image with Ubuntu and Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_ENDPOINT=https://hf-mirror.com  # For better connectivity in some regions

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

# Set workdir
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# Pre-download Hugging Face ZipVoice model using the correct command
RUN huggingface-cli download k2-fsa/ZipVoice zipvoice_distill.pt --local-dir /models/zipvoice

# Alternative if the above doesn't work:
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='k2-fsa/ZipVoice', allow_patterns='zipvoice_distill.pt', local_dir='/models/zipvoice')"

# Copy app code (after dependencies for better caching)
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]