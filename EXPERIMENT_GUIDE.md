# VideoSense Experiment Guide

Use this guide together with `notebooks/02_experiments.ipynb`. It covers:

1. How the files and runs depend on each other.
2. The order to run experiments in.
3. The hyperparameters for each part: what to set, why, and how to decide.
4. The rules that keep the reported numbers honest.
5. Where the evidence for each reflection question comes from.

> **About the rubric:** the PDF lists "12. Grading Rubric (Total: 100 Points)" in its table of contents, but the rubric itself was not in the document text. This guide follows the deliverables in Section 11 and every numbered exercise. Check the point weights with your instructor so you can spend time where the points are.

---

## 1. Dependency map

### 1.1 Code dependencies (who imports whom)

```
utils/dataset.py ───────────────┐   cached RGB frames -> (3,T,112,112) tensors
utils/optical_flow.py ──► utils/flow_dataset.py   cached flow -> (2,T,112,112), RGB+flow with shared augmentation
                                │
                                ▼
                         utils/common.py   splits, dataset building, training loop, checkpoints, test metrics
                                │
models/cnn3d.py ──► models/siamese.py      C3D, R(2+1)D, VideoEncoder ──► Siamese network
        └─────────► models/fusion.py       VideoEncoder ──► two-stream fusion
                         models/__init__.py   registry used to rebuild any model from its checkpoint
                                │
      ┌──────────────┬──────────┴──────┬────────────────┐
train_3dcnn.py  train_siamese.py  train_fusion.py   evaluate.py (+ utils/viz.py)
```

`utils/dataset.py`, `utils/optical_flow.py` and `utils/flow_dataset.py` are also written by the data-preparation notebook (Steps 3–4). The versions in this package add a `temporal_jitter` option for test-time augmentation. Keep this package's copies; if you re-run those notebook cells, the updated notebook in `notebooks/` writes the same versions.

### 1.2 Data and run dependencies (what must exist before what)

| To run… | You need first | Produced by |
|---|---|---|
| any training script | `splits_<subset>.csv`, `preprocess_config.json`, frame cache (and flow cache for fusion) | Notebook 01, Steps 1–4 |
| `train_3dcnn.py` (Part 2) | data only | — |
| `train_siamese.py --init-from` (optional) | a C3D trained **on base classes only** | `train_3dcnn.py --exclude-novel 5` |
| `train_fusion.py --mode attention/concat --init-rgb/--init-flow` (optional) | single-stream runs | `train_fusion.py --mode rgb` / `--mode flow` |
| `evaluate.py compare` (Ex 24) | RGB-only run + fusion run on the same subset | Part 2 or Part 4 runs |
| `evaluate.py tsne` (Ex 30) | a C3D run, a Siamese run, a fusion run | Parts 2, 3, 4 |
| `evaluate.py gradcam` / `report` / `tta` | any classifier or fusion run | Parts 2, 4 |
| `--subset full` runs | full frame + flow caches | Notebook 01 Steps 3.3 and 4.5 with `SUBSET='full'` |

---

## 2. Run order and compute budget

| Phase | Runs | Purpose |
|---|---|---|
| 0. Smoke test | every script with `--epochs 1 --max-batches 3` | catches path or cache errors in minutes rather than after an hour |
| 1. Part 2 | `c3d_ls0.1` → `c3d_ls0.0`, `c3d_ls0.2` → `r2p1d_head` (→ `r2p1d_layer4` optional) | baseline, label-smoothing sweep, transfer learning |
| 2. Part 3 | `c3d_base` → `siam_m1`, `siam_m2`, `siam_m4` → `siam_triplet` (→ `siam_bh`) | metric learning, margin study, loss comparison |
| 3. Part 4 | `fus_rgb`, `fus_flow`, `fus_concat` → `fus_attn` (→ `fus_concat_init`) | ablation, attention weight |
| 4. Part 5 | `evaluate.py` tasks | report figures |
| 5. Final | chosen configs with `--subset full` | headline numbers (if your compute allows) |

**Estimating time:** after the first epoch of any run, read `time_s` in `history.csv` and multiply by the number of epochs. The full dataset has roughly 10× as many clips as the mini subset, so a full run costs about 10× a mini run. Long runs survive disconnects with `--resume`, re-run with **identical** arguments.

**Drive space:** a C3D `best.pt` is about 250 MB (63M parameters, mostly the two 4096-unit FC layers). `last.pt`, which stores optimizer state, is deleted automatically when a run finishes unless you pass `--keep-last`. Delete smoke-test runs afterwards.

---

## 3. Hyperparameters: what, why, and how to decide

### 3.0 Principles that apply everywhere

- **Change one thing at a time** from a working baseline. Otherwise you can't tell which change caused the difference.
- **Select on validation, report on test.** Every script picks `best.pt` by a validation metric. When comparing configurations, read `best_val_*` in `evaluate.py summary`. Report `test_*` for the chosen configuration.
- **Develop on the mini subset** (minutes per run); spend full-dataset compute only on final configurations.
- **Read the curves before the numbers.** Train loss far below val loss means overfitting: more regularization or augmentation, or pretraining. Both stuck high means underfitting or an optimization problem (learning rate, initialization).
- **Differences smaller than the noise aren't findings.** On the mini test split (~40 clips per class), one or two percentage points can be noise. Rerun the top two configurations with another `--seed` before claiming a winner.

### 3.1 Part 2 — C3D from scratch (`train_3dcnn.py --model c3d`)

| Parameter | Default | Try | Intuition |
|---|---|---|---|
| `--lr` (Adam) | **3e-4** | 1e-4, 3e-4, (1e-3 to reproduce) | Adam moves *every* weight by roughly `lr` per step, whatever its gradient size. C3D's fc6/fc7 layers have 4608×4096 and 4096×4096 weights, so at 1e-3 these small coordinated moves add up and the logits explode. In testing, **loss went 2.3 → ~445 after one step at 1e-3** and fell smoothly at 3e-4 and 1e-4. The spike can kill ReLU units permanently. If epoch-1 loss is far above ln(C) (2.3 for 10 classes, 4.6 for 101), lower the lr. |
| `--epochs` | 30 | 20–40 | With cosine decay the lr reaches ~0 at the end, so more epochs give a longer high-lr phase, not just extra time. Stop adding epochs when val accuracy has plateaued for the last third. |
| `--label-smoothing` | 0.1 | 0.0, 0.1, 0.2 (Ex 8) | The target becomes 0.9 for the true class and 0.1 spread over the others. This stops the model from pushing logits toward infinity on a small dataset with near-duplicate clips. Expect higher *train* loss but often a smaller train–val gap. Too much (0.2+) can blur genuinely confusable classes. |
| `--dropout` | 0.5 | 0.3–0.6 | Regularizes the 4096-unit FC layers, which hold most of the parameters and most of the overfitting capacity. |
| `--weight-decay` | 1e-4 | 1e-4, 5e-4 (`--optimizer adamw`) | With Adam, `weight_decay` is an L2 penalty that gets rescaled by Adam. AdamW applies decay directly and is usually the better regularizer. |
| `--batch-size` | 8 | as large as memory allows | Bigger batches give smoother gradients. C3D without `--bn` has no BatchNorm, so small batches are fine. With `--bn`, batches under ~8 make the BatchNorm statistics noisy (Q13). |
| `--bn` | off | on | BatchNorm makes deep plain networks (C3D is VGG-style, 8 conv layers) much easier to optimize and often allows a higher lr. It's not in the original C3D, so report it as a modification if you use it. |

**Decision rule:** pick the label-smoothing value with the best `best_val_top1`. If two are within ~1 point, prefer the one with the smaller train–val gap.

### 3.2 Part 2 — Pretrained R(2+1)D-18 (`--model r2plus1d`)

| Parameter | Default | Try | Intuition |
|---|---|---|---|
| `--finetune` | `head` (Ex 9) | `layer4`, `all` | `head` = linear probe: it shows how good Kinetics features already are for UCF101. `layer4` adapts the most task-specific layers and is usually the best accuracy for the compute. `all` risks overfitting on the mini subset. |
| `--lr` | 3e-4 → use **1e-3** for `head` | 1e-3 (head), 1e-4 (layer4/all) | A freshly initialized head needs a normal lr; pretrained layers need small steps so they don't forget useful features. |
| `--backbone-lr-mult` | 1.0 | 0.1 with `layer4`/`all` | Gives the backbone 10× smaller steps than the new head ("discriminative learning rates"). |
| `--epochs` | — | 10 (Ex 9) | A linear probe converges fast. |
| normalization | Kinetics (automatic) | — | Pretrained filters expect the statistics they were trained on. |
| frozen BatchNorm | automatic | — | When the backbone is frozen it stays in eval mode, so its BatchNorm keeps the Kinetics statistics instead of re-estimating them from batches of 8. |

**What to expect:** the pretrained model wins by a large margin, mostly because UCF101 has only ~18 independent source videos per class (Step 2.6), far too few to learn general motion features from scratch (Q12).

### 3.3 Part 3 — Siamese network (`train_siamese.py`)

**Geometry first.** Embeddings are L2-normalized, so they live on a unit sphere and **every distance is between 0 and 2**. Two *random* unit vectors in 512 dimensions are almost orthogonal, so their distance is ≈ √2 ≈ 1.41. Training has to move same-class pairs well below that and different-class pairs up toward it.

| Parameter | Default | Try | Intuition |
|---|---|---|---|
| `--margin` (contrastive) | 1.0 | 1.0, 2.0, 4.0 (Ex 16) | A negative pair stops contributing once its distance exceeds the margin. **1.0:** negatives below 1 get pushed, which is well within reach. **2.0:** negatives must be exactly opposite on the sphere, so nearly all of them keep pushing. **4.0:** impossible (distances are at most 2), so every negative always pushes. The negative term then dominates the loss and works against pulling positives together. Expect the histograms to show this. |
| `--margin` (triplet) | 0.3 | 0.2–0.5 | A *relative* requirement: the negative must be farther than the positive by this much. Too large and the problem becomes unsatisfiable, like above. |
| `--loss` | contrastive | triplet, triplet_batchhard (Ex 17) | Contrastive loss fixes absolute distances; triplet loss only cares about ordering, which matches nearest-neighbour evaluation better. Batch-hard uses the most informative pair in every batch. |
| `--lr` | 1e-4 | 3e-5–3e-4 | Metric learning is less stable than classification, because the targets depend on other samples' embeddings. |
| `--init-from` | none | `runs/c3d_base` | Starting from features that already separate classes speeds up training a lot. **Use a C3D trained with `--exclude-novel 5`.** A C3D trained on all classes has already seen the "novel" classes, which inflates few-shot accuracy. |
| `--pos-fraction` | 0.5 | 0.1 = 90% negatives (Q19 experiment) | With 90% negatives, most of the gradient pushes things apart and positives rarely get pulled together. The model can lower the loss just by spreading everything out. |
| `--embed-dim` | 512 | 128, 256 | Smaller embeddings regularize and make distances more meaningful. |
| `--pairs-per-epoch` | 2000 | 2000–10000 | Compute per epoch. Triplet and batch-hard modes are scaled to encode the **same number of clips**, so loss comparisons are at equal compute. |
| `--n-novel` | 5 | 5 (mini), 20–30 (full) | On the mini subset, 5-way evaluation uses all 5 novel classes in every episode. On the full dataset you get genuinely varied episodes. |

**Model selection:** by validation AUC (how well distance separates same from different pairs on base-class val clips). **Report:** novel-class 5-way 1-shot and 5-shot accuracy, mean ± std over 200 episodes, plus the 95% CI half-width the script prints. Both `nn` (the assignment's protocol) and `proto` (class-mean prototypes) are reported. Prototypes are usually better at 5-shot because averaging cancels clip-specific noise. Queries are drawn from **different UCF101 groups** than the support clips, so a query can't be a near-duplicate of a support clip.

### 3.4 Part 4 — Two-stream fusion (`train_fusion.py`)

| Parameter | Default | Try | Intuition |
|---|---|---|---|
| `--mode` | concat | rgb, flow, concat (Ex 22); attention (Ex 23); avg, gated, cross_attn | All modes read identical (rgb, flow) batches with identical augmentation, so differences come only from the model. |
| Training strategy | joint from scratch | **`--init-rgb`/`--init-flow` from single-stream runs + `--backbone-lr-mult 0.1`** | Training both streams jointly often lets the faster-learning stream dominate while the other stays under-trained (Wang, Tran & Feiszli, CVPR 2020, *What Makes Training Multi-Modal Classification Networks Hard?*). Training each stream alone first, then fusing, usually beats joint training, and you need the single-stream runs for the ablation anyway. |
| `--freeze-encoders` | off | on with init | Cheapest fusion: learn only the fusion and classifier on top of two trained streams. A good first try when compute is tight. |
| `--camera-comp` | off | on | Subtracts per-frame median flow (Step 4.3). Most likely to help classes with strong camera motion; check per class. |
| `--lr` | 3e-4 | 1e-4–3e-4 | Same reasoning as C3D: each stream is a C3D. |
| `--embed-dim` | 512 | 256 | Size of each stream's embedding before fusion. |

**Interpreting α (Ex 23, Q27):** α is the weight on RGB. Both embeddings pass through LayerNorm before weighting; without it the network could make α meaningless by rescaling one embedding instead of changing α. α → 1 means flow adds little beyond RGB; α → 0 means motion dominates. Values near 0.5 can mean both are useful **or** that α barely moved, so check the `alpha_rgb` column across epochs. `gated` mode computes α per clip, and `test_outputs.npz` stores it, so you can see which classes lean on motion.

### 3.5 Part 5 — Evaluation settings

| Setting | Default | Intuition |
|---|---|---|
| Grad-CAM layer | last conv of the RGB stream | For C3D this layer is only **2 × 7 × 7** (time × height × width). Saliency is upsampled from 2 time steps, so it can't show frame-level timing. Say so in your observations. |
| t-SNE perplexity | min(30, n/4) | Roughly the number of neighbours each point "cares" about. With few points per class, lower is better. |
| Silhouette | on L2-normalized embeddings | t-SNE distorts global distances, so cluster quality is measured in the embedding space. Normalizing removes scale differences between models (4096-d C3D vs 512-d Siamese). The 2-D value is also printed, for reference only. |
| TTA crops | 5 | Averages predictions over different temporal samplings. Gains are typically small (≈0.5–2 points); larger gains suggest the model is sensitive to which frames it sees. |

---

## 4. Rules that keep results honest

1. **Test is evaluated once per run** (automatically, with `best.pt`) and never used to choose anything. Choose by validation.
2. **Validation is split by group** (groups 8–10), so near-duplicate clips never cross the split.
3. **Few-shot novel classes are never trained on**, including through `--init-from`. Use `c3d_base` for initialization.
4. **Report uncertainty:** few-shot mean ± std (and CI); for close classification results, a second seed.
5. **Report deviations from the template**: lr 3e-4, `--bn`, aspect-preserving crops, segment sampling, flow normalization. These are design decisions to explain in the Methods section, not things to hide.

---

## 5. Where each reflection answer gets its evidence

| Question | Evidence |
|---|---|
| Q1 fixed frame count | Notebook 01, Step 2.5 plots (sampling gap vs. coverage); segment sampling (Step 3.5) as the mitigation |
| Q2 camera vs. subject motion | Step 4.3 figure (median subtraction); per-class `camera_share` (Step 4.4); optional `--camera-comp` run |
| Q3 walking vs. running flow | Step 4.4 figures (shared speed scale, speed histograms) |
| Q11 C3D temporal receptive field | Shape trace in `models/cnn3d.py` docstring (16 → 1 time step after pool5); Grad-CAM's 2-step resolution |
| Q12 transfer learning | `c3d_ls*` vs `r2p1d_head` in `evaluate.py summary`; Step 2.6 (≈18 source videos per class) |
| Q13 BatchNorm with tiny batches | `--bn` C3D at `--batch-size 2` vs 8; alternatives: GroupNorm, frozen BN statistics (as in `r2p1d_head`) |
| Q18 L2 normalization | Section 3.3 geometry (bounded distances, margin meaning) |
| Q19 90% negatives | `--pos-fraction 0.1` run (share of *positive* pairs): distance histograms + few-shot accuracy |
| Q20 embedding geometry | `evaluate.py distances` (epoch 1 vs. best epoch), t-SNE |
| Q25 middle fusion | `cross_attn` mode fuses intermediate feature maps; discuss where to fuse |
| Q26 cheap motion at test time | Step 2.8 frame differences (~free) vs. Farneback cost (Step 4.1 timing) |
| Q27 α near 0 or 1 | `alpha_rgb` history + `mean_alpha_rgb` |
| Q33 background attention | Grad-CAM GIFs; remedies: person-centred crops, background-swap or cut-mix augmentation, flow stream |
| Q34 overlapping clusters | Triplet / batch-hard vs. contrastive; larger embedding margin; deeper projector |
| Q35 accuracy vs. silhouette | `silhouette_scores.csv` alongside test top-1 in `results_summary.csv` |
