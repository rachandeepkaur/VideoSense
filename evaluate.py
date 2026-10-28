"""Part 5: evaluation, visualization and interpretation (Exercises 7, 10, 16, 24, 28-32).

Tasks
-----
summary    : one results table for every run (val + test metrics, few-shot results)
curves     : train/val loss and accuracy curves for one or more runs                (Ex 7, 14, 21)
report     : top-1/top-5, per-class precision/recall, macro-F1, confusion matrix   (Ex 28, 29)
compare    : per-class accuracy of run B vs run A, top-5 gains                     (Ex 24)
tsne       : t-SNE of embeddings for several models + silhouette scores           (Ex 30)
gradcam    : Grad-CAM saliency GIFs and frame strips                              (Ex 31)
tta        : test-time augmentation with several temporal crops                    (Ex 32)
complexity : parameters, multiply-adds and latency of C3D vs R(2+1)D              (Ex 10)
distances  : Siamese distance histograms and the margin study                     (Ex 14, 16)

Examples
--------
python evaluate.py summary
python evaluate.py curves --runs <runs>/c3d_ls0.0 <runs>/c3d_ls0.1 <runs>/c3d_ls0.2 --out figs/ls_curves.png
python evaluate.py report --run <runs>/fus_concat
python evaluate.py tsne --runs <runs>/c3d_ls0.1 <runs>/siam_m1 <runs>/fus_concat --labels C3D Siamese Fusion
python evaluate.py gradcam --run <runs>/c3d_ls0.1 --n-clips 5
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import classification_report, f1_score, silhouette_score  # noqa: E402

from utils import common as U  # noqa: E402
from utils import viz  # noqa: E402
from utils.dataset import denormalize  # noqa: E402


def _name(run):
    return os.path.basename(os.path.normpath(run))


def _out(args, filename):
    os.makedirs(args.out_dir, exist_ok=True)
    return os.path.join(args.out_dir, filename)


def _dataset(meta, args, phase='test', classes=None, **kw):
    spec = dict(meta['data'])
    if classes is not None:
        spec['classes'] = sorted(classes)
    cfg = U.load_preprocess_config(args.meta_dir)
    frames = U.phase_frames(spec, args.meta_dir)
    return U.build_dataset(spec, frames[phase], args.data_root, cfg, train=False, **kw), frames[phase], spec


# ----------------------------------------------------------------------------- summary
def task_summary(args):
    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(args.runs_dir, '*', 'config.json'))):
        run = os.path.dirname(cfg_path)
        meta = json.load(open(cfg_path))
        a = meta.get('args', {})
        row = dict(run=_name(run), part=meta.get('part'), model=meta.get('model_type'),
                   mode=meta.get('model_kwargs', {}).get('mode', ''), subset=meta['data']['subset'],
                   n_classes=len(meta['data']['classes']), label_smoothing=a.get('label_smoothing'),
                   loss=a.get('loss'), margin=a.get('margin'), lr=a.get('lr'),
                   params_M=round(meta['params_total'] / 1e6, 2) if meta.get('params_total') else None)
        hist = os.path.join(run, 'history.csv')
        if os.path.exists(hist):
            h = pd.read_csv(hist)
            if 'val_top1' in h:
                row.update(best_val_top1=h.val_top1.max(), best_epoch=int(h.epoch[h.val_top1.idxmax()]))
            if 'val_auc' in h:
                row.update(best_val_auc=h.val_auc.max(), best_epoch=int(h.epoch[h.val_auc.idxmax()]))
            if 'alpha_rgb' in h:
                row['final_alpha_rgb'] = h.alpha_rgb.iloc[-1]
        tm = os.path.join(run, 'test_metrics.json')
        if os.path.exists(tm):
            t = json.load(open(tm))
            row.update(test_top1=t['top1'], test_top5=t['top5'], test_macro_f1=t['macro_f1'])
            if 'mean_alpha_rgb' in t:
                row['mean_alpha_rgb'] = t['mean_alpha_rgb']
        fs = os.path.join(run, 'fewshot.json')
        if os.path.exists(fs):
            for k, v in json.load(open(fs)).items():
                if isinstance(v, dict) and 'mean' in v:
                    row[k] = f'{v["mean"]:.3f}+/-{v["std"]:.3f}'
        rows.append(row)
    df = pd.DataFrame(rows).dropna(axis=1, how='all')
    if df.empty:
        print('No runs found in', args.runs_dir)
        return
    path = _out(args, 'results_summary.csv')
    df.to_csv(path, index=False)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.round(4).to_string(index=False))
    print('\nSaved', path, '\nSelect configurations by best_val_*; report test_* only for the chosen ones.')


# ----------------------------------------------------------------------------- curves
def task_curves(args):
    metrics = tuple(args.metrics.split(','))
    viz.plot_history(args.runs, args.labels, metrics, out=args.out or _out(args, 'curves.png'), title=args.title)


# ----------------------------------------------------------------------------- report
def task_report(args):
    meta = json.load(open(os.path.join(args.run, 'config.json')))
    names = meta['data']['classes']
    out = np.load(os.path.join(args.run, 'test_outputs.npz'))
    logits, labels = torch.tensor(out['logits']), out['labels']
    preds = logits.argmax(1).numpy()
    k = min(5, logits.shape[1])
    top5 = (logits.topk(k, 1).indices.numpy() == labels[:, None]).any(1).mean()
    print(f'{_name(args.run)} | Top-1 {np.mean(preds == labels):.4f} | Top-{k} {top5:.4f} | '
          f'macro-F1 {f1_score(labels, preds, average="macro"):.4f}\n')
    rep = classification_report(labels, preds, labels=range(len(names)), target_names=names,
                                digits=3, zero_division=0)
    print(rep)
    cm = viz.confusion(labels, preds, len(names))
    viz.plot_confusion_top(cm, names, args.top_n, out=_out(args, f'confusion_{_name(args.run)}.png'))
    pairs = viz.top_confused_pairs(cm, names, k=10)
    print('Most-confused pairs (true -> predicted, fraction of the true class):')
    for t, p_, r in pairs:
        print(f'  {t:>22s} -> {p_:<22s} {r:.2f}')
    with open(_out(args, f'report_{_name(args.run)}.txt'), 'w') as f:
        f.write(rep + '\nMost-confused pairs:\n' + '\n'.join(f'{t} -> {p_}: {r:.3f}' for t, p_, r in pairs))


# ----------------------------------------------------------------------------- compare
def task_compare(args):
    acc = [json.load(open(os.path.join(r, 'test_metrics.json')))['per_class_acc'] for r in (args.run_a, args.run_b)]
    la, lb = args.labels or [_name(args.run_a), _name(args.run_b)]
    _, gain = viz.plot_per_class_gain(acc[0], acc[1], la, lb, top=args.top,
                                      out=_out(args, f'per_class_{_name(args.run_b)}_vs_{_name(args.run_a)}.png'))
    print(f'Largest gains of {lb} over {la}:')
    print(gain.sort_values(ascending=False).head(args.top).round(3).to_string())
    print(f'\nLargest losses:\n{gain.sort_values().head(3).round(3).to_string()}')


# ----------------------------------------------------------------------------- t-SNE
@torch.no_grad()
def _embeddings(model, loader, device, max_batches=None):
    model.eval()
    embs, labels = [], []
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        *inputs, y = batch
        embs.append(model.get_embedding(*[x.to(device) for x in inputs]).float().cpu())
        labels.append(y)
    return torch.cat(embs).numpy(), torch.cat(labels).numpy()


def task_tsne(args):
    device = U.get_device()
    all_classes = sorted(U.load_splits(args.meta_dir, json.load(open(
        os.path.join(args.runs[0], 'config.json')))['data']['subset']).class_name.unique())
    classes = all_classes[:args.n_classes]
    labels_txt = args.labels or [_name(r) for r in args.runs]
    fig, axes = plt.subplots(1, len(args.runs), figsize=(6 * len(args.runs), 5.5))
    axes = np.atleast_1d(axes)
    scores = []
    for run, lab, ax in zip(args.runs, labels_txt, axes):
        model, meta = U.load_run(run, device)
        ds, df, _ = _dataset(meta, args, args.phase, classes=classes)
        loader = U.make_loader(ds, args.batch_size, False, args.num_workers)
        X, y = _embeddings(model, loader, device, args.max_batches)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)     # same scale for every model
        perp = min(30, max(5, len(X) // 4))
        X2 = TSNE(n_components=2, perplexity=perp, init='pca', random_state=args.seed).fit_transform(Xn)
        sil = silhouette_score(Xn, y, metric='euclidean') if len(set(y)) > 1 else float('nan')
        sil2d = silhouette_score(X2, y) if len(set(y)) > 1 else float('nan')
        scores.append(dict(model=lab, run=_name(run), dim=X.shape[1], n=len(X),
                           silhouette_embedding=sil, silhouette_tsne2d=sil2d))
        viz.plot_tsne(X2, y, classes, f'{lab}\nsilhouette (embedding) = {sil:.3f}', ax)
        seen = meta.get('novel_classes')
        if seen:
            ax.text(0.01, 0.01, f'novel (unseen) classes: {", ".join(c for c in seen if c in classes)}',
                    transform=ax.transAxes, fontsize=6, va='bottom')
    axes[-1].legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle(f't-SNE of {args.phase} embeddings ({len(classes)} classes)')
    fig.tight_layout()
    path = _out(args, 'tsne_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    df = pd.DataFrame(scores)
    df.to_csv(_out(args, 'silhouette_scores.csv'), index=False)
    print(df.round(4).to_string(index=False))
    print('\nSilhouette is computed on the L2-normalized embeddings, not on the 2-D t-SNE map '
          '(t-SNE distorts global distances).\nSaved', path)


# ----------------------------------------------------------------------------- Grad-CAM
def task_gradcam(args):
    device = U.get_device()
    model, meta = U.load_run(args.run, device)
    if meta['model_type'] == 'siamese':
        raise SystemExit('Grad-CAM needs class scores: use a classifier or fusion run.')
    ds, df, spec = _dataset(meta, args, args.phase)
    rgb_ds = ds.rgb if hasattr(ds, 'rgb') else ds
    names = spec['classes']
    picks = df.drop_duplicates('label').index[:args.n_clips].tolist()        # one clip per class
    cam_tool = viz.VideoGradCAM(model, model.cam_layer)
    show_t = np.linspace(0, U.load_preprocess_config(args.meta_dir)['T'] - 1, 4, dtype=int)
    fig, axes = plt.subplots(len(picks), len(show_t), figsize=(3 * len(show_t), 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        sample = ds[i]
        inputs = [x.unsqueeze(0).to(device) for x in sample[:-1]]
        cam, pred, prob, native = cam_tool(inputs)
        frames = denormalize(rgb_ds[i][0], spec['norm'])
        over = viz.overlay_cam(frames, cam)
        tag = f'{df.class_name[i]}_{r}'
        viz.save_gif(over, _out(args, f'gradcam_{_name(args.run)}_{tag}.gif'), fps=args.fps)
        for c, t in enumerate(show_t):
            axes[r, c].imshow(over[t])
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].set_title(f'true: {df.class_name[i]}\npred: {names[pred]} ({prob:.2f})', fontsize=8)
            else:
                axes[r, c].set_title(f't={t}', fontsize=8)
    cam_tool.remove()
    fig.suptitle(f'Grad-CAM ({_name(args.run)}); CAM layer resolution (t, h, w) = {native}')
    fig.tight_layout()
    path = _out(args, f'gradcam_{_name(args.run)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {len(picks)} GIFs and {path}. Native CAM resolution {native}: the map is upsampled '
          'to 16x112x112, so temporal detail is limited to that many time steps.')


# ----------------------------------------------------------------------------- TTA
def task_tta(args):
    device = U.get_device()
    model, meta = U.load_run(args.run, device)
    names = meta['data']['classes']
    base_ds, _, _ = _dataset(meta, args, 'test')
    base = U.run_epoch(model, U.make_loader(base_ds, args.batch_size, False, args.num_workers), device,
                       nn.CrossEntropyLoss(), max_batches=args.max_batches, collect=True)
    summed = torch.zeros_like(base['logits'])
    for crop in range(args.n_crops):
        torch.manual_seed(1000 + crop)                    # reproducible temporal jitter per pass
        random.seed(1000 + crop)
        ds, _, _ = _dataset(meta, args, 'test', temporal_jitter=True)
        r = U.run_epoch(model, U.make_loader(ds, args.batch_size, False, args.num_workers), device,
                        nn.CrossEntropyLoss(), max_batches=args.max_batches, collect=True)
        summed += r['logits']
    labels = base['labels']
    k = min(5, len(names))
    tta1 = (summed.argmax(1) == labels).float().mean().item()
    tta5 = (summed.topk(k, 1).indices == labels[:, None]).any(1).float().mean().item()
    res = dict(single_top1=base['top1'], single_top5=base['top5'], tta_top1=tta1, tta_top5=tta5,
               n_crops=args.n_crops)
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.run, 'tta_metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)


# ----------------------------------------------------------------------------- complexity
def task_complexity(args):
    import time
    from torchinfo import summary
    from models import build_model
    device = U.get_device()
    n = args.num_classes
    candidates = {
        'C3D (scratch)': ('c3d', dict(num_classes=n)),
        'R(2+1)D-18 (head only)': ('r2plus1d', dict(num_classes=n, pretrained=False, freeze_backbone=True)),
        'R(2+1)D-18 (full fine-tune)': ('r2plus1d', dict(num_classes=n, pretrained=False, freeze_backbone=False)),
    }
    rows = []
    for label, (mtype, kw) in candidates.items():
        model = build_model(mtype, **kw).to(device).eval()
        s = summary(model, input_size=(1, 3, 16, 112, 112), verbose=0, device=device)
        x = torch.randn(1, 3, 16, 112, 112, device=device)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.reps):
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        rows.append(dict(model=label, params_M=s.total_params / 1e6, trainable_M=s.trainable_params / 1e6,
                         GMACs=s.total_mult_adds / 1e9, approx_GFLOPs=2 * s.total_mult_adds / 1e9,
                         latency_ms=(time.time() - t0) / args.reps * 1000, device=device.type))
    df = pd.DataFrame(rows)
    df.to_csv(_out(args, 'complexity.csv'), index=False)
    print(df.round(2).to_string(index=False))
    print('\nFLOPs ~ 2 x multiply-adds. Latency: batch 1, 16x112x112 clip.')


# ----------------------------------------------------------------------------- distances
def task_distances(args):
    labels_txt = args.labels or [_name(r) for r in args.runs]
    fig, axes = plt.subplots(2, len(args.runs), figsize=(5 * len(args.runs), 7), squeeze=False)
    for c, (run, lab) in enumerate(zip(args.runs, labels_txt)):
        meta = json.load(open(os.path.join(run, 'config.json')))
        margin = meta['args'].get('margin') if meta['args'].get('loss') == 'contrastive' else None
        files = sorted(glob.glob(os.path.join(run, 'distances', 'epoch_*.npz')))
        h = pd.read_csv(os.path.join(run, 'history.csv'))
        best = int(h.epoch[h.val_auc.idxmax()])
        for r, (f, when) in enumerate([(files[0], 'epoch 1'), (files[best - 1], f'best epoch {best}')]):
            d = np.load(f)
            viz.plot_distance_hist(d['pos'], d['neg'], f'{lab}: {when}', axes[r, c], margin)
    fig.suptitle('Validation pair distances (same vs different class)')
    fig.tight_layout()
    path = _out(args, 'siamese_distance_histograms.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    viz.plot_history(args.runs, labels_txt, ('loss', 'auc'), out=_out(args, 'siamese_curves.png'))
    print('Saved', path)


def main():
    p = argparse.ArgumentParser(description='VideoSense evaluation (Part 5)')
    p.add_argument('task', choices=['summary', 'curves', 'report', 'compare', 'tsne', 'gradcam', 'tta',
                                    'complexity', 'distances'])
    p.add_argument('--data-root', default=U.COLAB_DATA_ROOT)
    p.add_argument('--meta-dir', default=U.COLAB_META_DIR)
    p.add_argument('--runs-dir', default=U.COLAB_RUNS_DIR)
    p.add_argument('--out-dir', default='/content/drive/MyDrive/videosense/figures/results')
    p.add_argument('--run')
    p.add_argument('--runs', nargs='+')
    p.add_argument('--run-a')
    p.add_argument('--run-b')
    p.add_argument('--labels', nargs='+')
    p.add_argument('--title')
    p.add_argument('--out')
    p.add_argument('--metrics', default='loss,top1')
    p.add_argument('--phase', default='test', choices=['val', 'test'])
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--n-classes', type=int, default=20)
    p.add_argument('--n-clips', type=int, default=5)
    p.add_argument('--n-crops', type=int, default=5)
    p.add_argument('--num-classes', type=int, default=101)
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    globals()[f'task_{args.task}'](args)


if __name__ == '__main__':
    main()
