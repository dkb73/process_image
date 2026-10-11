# Run steps (this studio)

## 1. Run setup.sh (apt packages + Triton C++ Client SDK + Python basics)

```bash
cd ~/process_image
chmod +x setup.sh
sudo ./setup.sh
```

## 2. Install libs and dependencies

```bash
conda install -y -c pytorch -c nvidia faiss-gpu

pip install onnx onnxruntime numpy opencv-python insightface ultralytics
```

## 3. Download real pre-trained ONNX models

```bash
python3 tools/download_models.py --real --gpu
```

## 4. Start Triton server

```bash
docker run --gpus all -d --name triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models \
      --backend-config=onnxruntime,default-max-batch-size=8
```

## 5. Check if the server is running

```bash
curl -s -o /dev/null -w "live %{http_code}\n" localhost:8000/v2/health/live
curl -s -o /dev/null -w "ready %{http_code}\n" localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/yolo26_face | python3 -m json.tool
curl -s localhost:8000/v2/models/arcface | python3 -m json.tool
```

`ready` must be `200` before continuing.

## 6. Build

```bash
cd ~/process_image
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 7. Install the Triton Python client

```bash
pip install "tritonclient[grpc]"
```

## 8. Warm ArcFace batch sizes

Run this after every Triton container start, before any video. It sends one zero tensor per batch size `1..32` straight to the `arcface` model. The first pass is slow. Later frames with those face counts skip the cuDNN search. The cache dies when the container stops.

```bash
cd ~/process_image
python3 tools/warmup_arcface.py --triton localhost:8001
```

## 9. Enroll faces into the database

`faces/` must contain one subfolder per person.

```bash
python3 tools/enroll_faces.py \
    --faces-dir ./faces/ \
    --db ./faces.db \
    --triton localhost:8001
```

## Metrics-only run

After steps 1–9, this writes per-frame metrics and a detections CSV. Skip steps 10 and 11 when you only want those files.

```bash
./build/process_image \
  --input ./input/input_3.mp4 \
  --triton localhost:8001 \
  --db ./faces.db \
  --device auto
```

## 10. Run the pipeline

```bash
./build/video_analytics_service \
  --triton localhost:8001 \
  --db ./faces.db \
  --port 50051 \
  --max-sessions 30 \
  --device auto
```

## 11. Streamlit server

```bash
cd ~/process_image
pip install -r client/requirements.txt
streamlit run --server.port 8501 --server.address 0.0.0.0 client/app.py
```

Open:

```text
https://<port>-<LIGHTNING_CLOUD_SPACE_ID>.cloudspaces.litng.ai
```

Replace `<port>` with `8501`. Replace `<LIGHTNING_CLOUD_SPACE_ID>` with the output of:

```bash
echo "$LIGHTNING_CLOUD_SPACE_ID"
```

## After a studio or container restart

Triton and the ArcFace cache do not survive a new container. Repeat from step 4, then step 5, then step 8, before step 9 or a metrics run. Steps 1–3 and 6–7 can be skipped when the build and packages are already there.
