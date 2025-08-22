# Use NVIDIA CUDA base image with Ubuntu and Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/tmp/huggingface_cache
ENV NLTK_DATA=/usr/share/nltk_data

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

# Prefetch required Hugging Face assets into cache for zipvoice_distill
# This ensures runtime hf_hub_download calls find them without network
RUN python3 -c "from huggingface_hub import hf_hub_download; repo='k2-fsa/ZipVoice'; model_dir='zipvoice_distill'; [hf_hub_download(repo, filename=f'{model_dir}/{fname}') for fname in ['model.pt','model.json','tokens.txt','text_encoder.onnx','fm_decoder.onnx']]"

# spaCy English model (small)
RUN python3 -m spacy download en_core_web_sm

# NLTK data for g2p-en and potentially h2p-parser (pre-download for offline runtime)
RUN python3 -m nltk.downloader -d /usr/share/nltk_data averaged_perceptron_tagger_eng averaged_perceptron_tagger punkt cmudict

# Copy app code (after dependencies for better caching)
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]