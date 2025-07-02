FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Pre-download the model checkpoints
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='franciszzj/Leffa', \
    local_dir='./ckpts', \
    allow_patterns=['densepose/*', 'examples/*', 'humanparsing/*', 'openpose/*', 'schp/*', 'stable-diffusion-inpainting/*', 'virtual_tryon.pth', 'virtual_tryon_dc.pth'], \
    ignore_patterns=['pose_transfer.pth', 'stable-diffusion-xl*'])"

# Make temp directory for file operations
RUN mkdir -p /app/temp
ENV TEMP_DIR=/app/temp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Create the RunPod handler file
COPY handler.py .
COPY concurrent_handler.py .

# Set the entrypoint
CMD ["python", "concurrent_handler.py"]
