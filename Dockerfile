# Base image required by the spec.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps (ffmpeg + OpenCV runtime libs).
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Required directory layout.
RUN mkdir -p /app/data/input /app/data/output /app/weights /app/src

# Install deps at BUILD time (runtime has no internet).
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model weights + source (selective COPY to keep image under 8 GB).
# weights/ holds the VLM from_pretrained directory at
# weights/vlm/{config.json, *.safetensors, processor, ...} —
# NOT a single best_model.pt file.
COPY weights/ /app/weights/
COPY src/ /app/src/
COPY main.py .
COPY README.md .

# Auto-start on `docker run`.
CMD ["python3", "main.py"]
