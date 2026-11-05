# gpu_server image: CoTracker3 + RoMa v2 on one CUDA GPU (Linux, NVIDIA Container Toolkit).
# Build from the REPO ROOT (the root .dockerignore keeps the context to the files below):
#   docker build -f gpu_server/docker/Dockerfile -t vlm-gpu-server .
# Run: see gpu_server/docker/docker-compose.yml and gpu_server/README.md "Linux GPU box (Docker)".
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# NVIDIA_DISABLE_REQUIRE: the host driver may report an older CUDA (e.g. 535 = 12.2); torch's cu12x
# wheels run on any driver >= 525 via CUDA minor-version compatibility, so skip the version gate.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/huggingface \
    ROMAV2_ROOT=/opt/RoMaV2 \
    NVIDIA_DISABLE_REQUIRE=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY gpu_server/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# RoMa v2 sources only: `pip install -e` would pull a different torch (its pyproject pins torchvision).
ARG ROMAV2_REF=main
RUN git clone --depth 1 --branch ${ROMAV2_REF} https://github.com/Parskatt/RoMaV2 ${ROMAV2_ROOT}

WORKDIR /app
COPY config.py ./
COPY providers ./providers
COPY gpu_server ./gpu_server

# Weights (CoTracker hub repo + checkpoints, RoMa v2, HF models) are downloaded on first load here.
VOLUME ["/cache"]
EXPOSE 8765 8766 8775 8776

ENTRYPOINT ["python", "-m", "gpu_server.server"]
CMD ["--modules", "cotracker", "echo", "--preload", "cotracker", "--ws", "0.0.0.0:8765", "--tcp", "0.0.0.0:8766"]
