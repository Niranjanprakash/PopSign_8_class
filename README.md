# LightMamba-ASL PopSign 4-Class Recognition

This deployment recognizes eight ASL signs from a complete MP4 video or webcam capture:

- after
- airplane
- bird
- cloud
- cry
- dog
- drink
- elephant

The shipped checkpoint is `checkpoints/best_model.pth`. It was trained on PopSign videos with MobileNetV3-Large RGB features, MediaPipe Holistic landmarks, motion features, reliability-aware fusion, and the hierarchical temporal model.

## Evaluation

The best checkpoint achieved the following metrics on the untouched PopSign test split:

- Top-1 accuracy: 86.14%
- Macro F1 score: 0.8346
- Best epoch: 19
- Average inference latency on a T4 GPU: 24.82 ms
- Approximate T4 GPU throughput: 40.29 FPS

Per-class test accuracy:

| Sign | Accuracy |
| --- | ---: |
| after | 93.08% |
| airplane | 68.87% |
| bird | 88.59% |
| cloud | 80.00% |
| cry | 94.16% |
| dog | 84.21% |
| drink | 92.76% |
| elephant | 90.70% |

## Run Locally

Create and activate the virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the Flask backend:

```powershell
.\.venv\Scripts\python.exe app.py
```

In a second terminal, start the React frontend:

```powershell
cd lightmamba-asl-frontend
npm install
npm start
```

Open `http://localhost:3000`. Use Video Recognition to upload an MP4 or Live Recognition for a webcam demonstration.

## Test A Video

```powershell
.\.venv\Scripts\python.exe -m backend.inference.predict_video --video "C:\path\to\bird.mp4"
```

Only the eight supported signs should be used for reliable predictions. Other signs may be returned as one of these eight classes or marked uncertain.

## Optional Retraining Data

The local PopSign data is stored in `dataset/popsign` with this layout:

```text
dataset/popsign/
  train/<sign>/*.mp4
  val/<sign>/*.mp4
  test/<sign>/*.mp4
```

To prepare and retrain, set `POPSIGN_ROOT` to this folder before running the preparation, feature extraction, training, and evaluation modules.
