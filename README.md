# VideoSense: Multimodal Video Intelligence for Human Activity Recognition

DATA 255 Extra Credit HW3: 3D CNNs, Siamese networks and two-stream RGB + optical-flow fusion on UCF101 (split 1), in PyTorch.

**Read [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) before training.** It covers the dependency map, the run order, and the reasoning behind every hyperparameter.

## Project structure

```
videosense/
├── notebooks/
│   ├── 01_data_preparation.ipynb   Steps 1-4: download, EDA, splits, frame cache, optical flow
│   └── 02_experiments.ipynb        Steps 5-12: every training / evaluation command, in order
├── models/
│   ├── cnn3d.py        C3D, R(2+1)D block + network, pretrained R(2+1)D-18, shared VideoEncoder
│   ├── siamese.py      Siamese network, contrastive / triplet / batch-hard losses, samplers, few-shot eval
│   ├── fusion.py       two-stream model: rgb | flow | concat | avg | attention | gated | cross_attn
│   └── __init__.py     model registry (rebuilds any model from its checkpoint)
├── utils/
│   ├── dataset.py      cached-frame dataset, segment sampling, clip-consistent augmentation
│   ├── optical_flow.py Farneback flow, quantized JPEG storage, HSV visualization
│   ├── flow_dataset.py flow dataset + RGB/flow dataset with shared augmentation
│   ├── common.py       splits, data loaders, training loop, checkpoints, test metrics
│   └── viz.py          curves, confusion matrix, t-SNE, distance histograms, Grad-CAM, GIFs
├── train_3dcnn.py      Part 2 (Exercises 6-10)
├── train_siamese.py    Part 3 (Exercises 14-17)
├── train_fusion.py     Part 4 (Exercises 21-24)
├── evaluate.py         Part 5 (Exercises 28-32) + results summary, complexity, margin study
├── requirements.txt
├── README.md
└── EXPERIMENT_GUIDE.md
```

## Setup (Google Colab)

1. Upload this folder to Google Drive as `MyDrive/videosense/code/`.
2. Run `notebooks/01_data_preparation.ipynb` on a **CPU** runtime (Steps 1-4). It downloads UCF101, builds the splits, and writes the frame and flow caches. Each cache is backed up to Drive as a single `.tar`.
3. Switch to a **GPU** runtime and open `notebooks/02_experiments.ipynb`. Its first cells mount Drive, install `torchinfo` and `imageio`, and restore the caches to local disk.

Locally, install the dependencies with `pip install -r requirements.txt` and pass `--data-root`, `--meta-dir` and `--runs-dir` to every script (the defaults are the Colab paths).

## Running

All scripts share these options: `--subset mini|full`, `--run-name`, `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--weight-decay`, `--seed`, `--resume`, `--trainval`, `--export`, `--max-batches` (smoke test), `--keep-last`. Run any script with `--help` to see its full option list.

```bash
# Part 2
python train_3dcnn.py   --run-name c3d_ls0.1  --model c3d --label-smoothing 0.1 --epochs 30
python train_3dcnn.py   --run-name r2p1d_head --model r2plus1d --finetune head --epochs 10 --lr 1e-3
# Part 3
python train_3dcnn.py   --run-name c3d_base   --model c3d --exclude-novel 5
python train_siamese.py --run-name siam_m1    --loss contrastive --margin 1.0 --init-from <runs>/c3d_base
python train_siamese.py --run-name siam_triplet --loss triplet --margin 0.3 --init-from <runs>/c3d_base
# Part 4
python train_fusion.py  --run-name fus_rgb    --mode rgb
python train_fusion.py  --run-name fus_flow   --mode flow
python train_fusion.py  --run-name fus_concat --mode concat
python train_fusion.py  --run-name fus_attn   --mode attention --init-rgb <runs>/fus_rgb --init-flow <runs>/fus_flow
# Part 5
python evaluate.py summary
python evaluate.py report  --run <runs>/fus_concat
python evaluate.py tsne    --runs <runs>/c3d_ls0.1 <runs>/siam_m1 <runs>/fus_concat --labels C3D Siamese Fusion
python evaluate.py gradcam --run <runs>/c3d_ls0.1 --n-clips 5
```

## What every run writes (`<runs>/<run-name>/`)

| File | Contents |
|---|---|
| `config.json` | model type and arguments, data specification (classes, normalization), all CLI arguments |
| `history.csv` | per-epoch train/val loss, top-1, top-5, lr, time (+ `alpha_rgb` for attention fusion; AUC and distances for Siamese) |
| `best.pt` | checkpoint with the best validation metric (model + metadata) |
| `last.pt` | full state for `--resume`; deleted when the run finishes unless `--keep-last` |
| `test_metrics.json` | top-1, top-5, macro-F1, per-class accuracy on the test split (evaluated once, with `best.pt`) |
| `test_outputs.npz` | test logits, labels, clip paths (used by `evaluate.py report` / `compare`) |
| `fewshot.json` | Siamese: novel-class N-way K-shot accuracy, mean / std / 95% CI |
| `distances/` | Siamese: validation pair distances per epoch (histograms for Exercise 16) |

**Resuming after a disconnect:** re-run the *same* command with `--resume` added.

## Checkpoints for submission (Section 11.2)

Add `--export` to the final runs (or call `utils.common.export_state_dict` afterwards). This writes plain `state_dict` files:

```bash
python train_3dcnn.py   ... --export models/c3d_ucf101.pt
python train_siamese.py ... --export models/siamese_ucf101.pt
python train_fusion.py  ... --export models/twostream_ucf101.pt
```

To load one, build the model with the arguments in that run's `config.json` (`model_kwargs`), then call `load_state_dict`.

## Design decisions that differ from the assignment template

| Template | This implementation | Why |
|---|---|---|
| Frames resized to 112×112 | Resize to 128×171, then random / center crop 112 | keeps the 4:3 aspect ratio and adds crop augmentation (C3D paper protocol) |
| Seek-based frame reads every epoch | Decode once, cache 32 frames per clip as JPEG | much faster loading, exact frames, fits on a free Drive |
| 16 uniformly spaced frames | Segment sampling (random frame per segment for train, fixed for eval) | temporal augmentation while keeping full coverage |
| Test set used as validation | Validation = held-out groups 8-10 of train; test evaluated once | no tuning on the test set; no near-duplicate leakage |
| Flow normalized per clip by its max | Flow divided by a fixed global bound | keeps absolute speed information |
| Adam lr 1e-3 for C3D | 3e-4 | 1e-3 makes the loss spike (2.3 → ~445) after one step |
| `register_backward_hook` Grad-CAM | tensor hook on the layer output | the old hook is deprecated and can return wrong gradients |
| Few-shot episodes re-encode clips | Embeddings computed once; queries from different groups than supports | about 100× faster; no near-duplicate support/query pairs |

## Code quality

Check style with `flake8 --max-line-length 120 .`.
