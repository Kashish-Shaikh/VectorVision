# Vector Vision

An explainable AI and drone system for finding mosquito breeding sites, built as an IRIS
**Robotics & Intelligent Machines** project. Every model is trained from random initialisation
on real, open data. No pretrained weights, no generated datasets, no hand-set risk scores.

```
Satellite + climate + terrain  ──►  Risk model (Random Forest, SHAP)   MACRO SCANNING
          ▲                                  │
Satellite U-Net ─ land cover fractions ──────┘
                                             ▼
Drone video ──► Drone U-Net ─ standing water per frame                 MICRO SCANNING
                                             ▼
             Priority Index = P(suitable habitat) × P(standing water)  THE BRAIN
                    + SHAP reasons for every target
                                             ▼
             GPS targets ─► waypoint mission ─► flight + drop simulation   THE STRIKE
```

## Setup in VS Code (once)

1. Open this folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Terminal → New Terminal**) and create an environment:

   **Windows**
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```
   **macOS / Linux**
   ```
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Install PyTorch first, matching your machine, from https://pytorch.org/get-started/locally
   (choose the CUDA build if you have an NVIDIA GPU). Then:
   ```
   pip install -r requirements.txt
   ```
4. Select the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
5. Earth Engine, once: `earthengine authenticate`

## Put the dataset in place

Copy `SEN-2 LULC.zip` (2.28 GB, from your Drive) into `data/satellite/sen2lulc/`.
Stage 1 extracts it for you. See `data/README.md` for every dataset and its official source.

## Run

```
python main.py stages     # list the stages
python main.py verify     # Stage 1
```
Or use the **Run and Debug** panel, which has a configuration for each stage.

## Stages

| # | Command | What it does | Needs GPU |
|---|---|---|---|
| 1 | `verify` | Environment + dataset verification | no |
| 2 | `train-lulc` | Satellite U-Net from scratch | yes |
| 3 | `eval-lulc` | Held-out test, per-class IoU | no |
| 4 | `build-features` | 100 m grid over Gondiya + vector-occurrence training set | no |
| 5 | `train-risk` | Random Forest, spatial cross-validation, SHAP | no |
| 6 | `train-drone` | Drone water U-Net from scratch on FloodNet | yes |
| 7 | `infer-video` | Frame-by-frame detection on MP4 | helps |
| 8 | `fuse` | Priority Index, ranked GPS targets | no |
| 9 | `mission` | Waypoints + flight and drop simulation | no |
| 10 | `dashboard` | Web dashboard | no |

Stages are added one at a time, each only after the previous one works on your machine.

No GPU? Run stages 2 and 6 in Colab with the same code:
upload this folder to Drive, then `!cd /content/drive/MyDrive/VectorVision && python main.py train-lulc`.

## Honesty rules this project follows

- No metric is written anywhere until the model has been trained and tested.
- Ground truth, proxy labels and predictions are always named as such.
- Test data is never used for training or model selection.
- Physical tests use artificial puddles and water-only payloads. No pesticides or larvicides.
