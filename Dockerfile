# Dockerfile for Stable Diffusion LoRA Fine-Tuning

# Use NVIDIA PyTorch base image with CUDA support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    gedit \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install diffusers transformers datasets accelerate xformers controlnet_aux bitsandbytes matplotlib torchmetrics torch-fidelity


