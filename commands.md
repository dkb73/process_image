# Setup & Run Commands (GPU Cluster - Ubuntu 24.04)

The default CMake build targets **NVIDIA T4** (Turing, SM 7.5). For other GPUs, override `CMAKE_CUDA_ARCHITECTURES` at build time (see [§8](#8-build-the-project)). Triton, models, and runtime flags are otherwise the same.

## 1. Verify GPU

```bash
nvidia-smi
```

## 2. Run setup.sh (apt packages + Triton C++ Client SDK + Python basics)

```bash
cd ~/process_image
chmod +x setup.sh
sudo ./setup.sh
```

## 3. Install NVIDIA Container Toolkit (for Docker GPU support)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 4. Install FAISS-GPU

```bash
# Option A: via conda
conda install -y -c pytorch -c nvidia faiss-gpu

# Option B: via pip (CPU-only fallback)
pip3 install --break-system-packages faiss-cpu
```

## 5. Install Python dependencies for model download

```bash
pip3 install --break-system-packages onnx onnxruntime numpy opencv-python insightface ultralytics
```

## 6. Download real pre-trained ONNX models

```bash
cd ~/process_image
python3 tools/download_models.py --real --gpu
```

This creates:
```
models/model_repository/
├── yolo26_face/         # YOLO face detection  [N,3,640,640] → [N,8400,5]
│   ├── config.pbtxt
│   └── 1/model.onnx
└── arcface/             # ArcFace recognition  [N,3,112,112] → [N,512]
    ├── config.pbtxt
    └── 1/model.onnx
```

## 7. Start Triton Inference Server (Docker, GPU)

```bash
cd ~/process_image

docker run --gpus all -d --name triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

Wait ~20 seconds, then verify:

```bash
curl -s localhost:8000/v2/health/live
curl -s localhost:8000/v2/models | python3 -m json.tool
```

## 8. Build the project

CUDA kernels in this repo are compiled for a specific **compute capability** (`CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`). The default is **75** (T4). Building with the wrong value can fail at link time or produce binaries that do not run correctly on your GPU.

### GPU → CUDA architecture

| GPU | Architecture | `CMAKE_CUDA_ARCHITECTURES` |
|-----|--------------|----------------------------|
| NVIDIA T4 | Turing | `75` (default) |
| RTX 30-series (3060, 3070, 3080, 3090) | Ampere | **`86`** |
| A100 | Ampere | `80` |
| A10, RTX A-series (e.g. A5000) | Ampere | `86` |
| RTX 40-series, L4 | Ada Lovelace | `89` |
| H100 | Hopper | `90` |

Check your GPU’s compute capability: `nvidia-smi --query-gpu=name,compute_cap --format=csv` or the [CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

### T4 (default)

```bash
cd ~/process_image

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### NVIDIA RTX 30-series (Ampere, SM 8.6)

Use this when building on or for RTX 3060 / 3070 / 3080 / 3090 (and similar Ampere consumer cards). Steps 1–7 above are unchanged; only the CMake architecture flag differs.

```bash
cd ~/process_image

# Clean rebuild if you previously built for T4 (sm_75)
rm -rf build

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)
```

**Notes for 30-series:**

- Install **FAISS-GPU** (conda `faiss-gpu`), not CPU-only FAISS, so matching runs on the GPU.
- Consumer cards often have **less VRAM than T4 (16 GB)**; reduce concurrent `video_analytics_service` sessions or per-frame face limits if you see CUDA OOM.
- Latency is typically **at or below** the T4 ~13 ms target; no code changes are required beyond the correct `CMAKE_CUDA_ARCHITECTURES`.

**Build on the same machine as the GPU** (optional):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

Use an explicit value (`86`, `75`, etc.) when the build host GPU differs from the deployment GPU.

### Other GPUs (one-liner)

```bash
# A100
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80

# RTX 40-series / L4
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89

# H100
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
```

## 9. Enroll faces into the database

```bash
cd ~/process_image

# faces/ directory should have subfolders per person:
#   faces/alice/photo1.jpg
#   faces/bob/photo1.jpg photo2.jpg
#   ...

python3 tools/enroll_faces.py \
    --faces-dir ./faces/ \
    --db ./faces.db \
    --triton localhost:8001
```

## 10. Run the pipeline

```bash
cd ~/process_image

./build/process_image \
    --input /path/to/video.mp4 \
    --triton localhost:8001 \
    --db ./faces.db \
    --config config/pipeline.yaml \
    --bbox-csv ./output/detections.csv
```

### All CLI options

```
--input <path|url>        MP4 file path or RTSP URL (required)
--triton <url>            Triton server URL (default: localhost:8001)
--config <path>           YAML config file (default: config/pipeline.yaml)
--db <path>               SQLite face database (default: ./faces.db)
--output-csv <path>       Metrics CSV path (default: ./metrics/metrics.csv)
--bbox-csv <path>         Detection CSV path (default: ./output/detections.csv)
--confidence <float>      YOLO confidence threshold (default: 0.5)
--match-threshold <float> Face similarity threshold (default: 0.6)
--device <gpu|cpu|auto>   Force device mode (default: auto)
--no-console              Suppress per-frame console output
```

## Cleanup

```bash
# Stop Triton
docker stop triton && docker rm triton

# Clean build
rm -rf build/
```

## Media server 

docker run --rm \
  --name mediamtx \
  --network host \
  bluenviron/mediamtx:1
