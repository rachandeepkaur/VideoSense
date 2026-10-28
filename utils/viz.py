"""Visualization utilities: training curves, confusion matrix, t-SNE, distance histograms,
per-class comparisons, and Grad-CAM for video (Part 5)."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.rcParams['figure.dpi'] = 110


def _save(fig, out):
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print('saved', out)


# ----------------------------------------------------------------------------- curves
def plot_history(run_dirs, labels=None, metrics=('loss', 'top1'), out=None, title=None):
    """Train (dashed) vs validation (solid) curves for one or more runs."""
    labels = labels or [os.path.basename(os.path.normpath(r)) for r in run_dirs]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.arange(len(run_dirs)) % 10)
    for run, lab, col in zip(run_dirs, labels, colors):
        h = pd.read_csv(os.path.join(run, 'history.csv'))
        for ax, m in zip(axes, metrics):
            if f'train_{m}' in h:
                ax.plot(h.epoch, h[f'train_{m}'], ls='--', color=col, alpha=0.7)
            if f'val_{m}' in h:
                ax.plot(h.epoch, h[f'val_{m}'], color=col, label=lab)
    for ax, m in zip(axes, metrics):
        ax.set(xlabel='epoch', ylabel=m, title=f'{m} (dashed = train, solid = val)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out)
    return fig


# ----------------------------------------------------------------------------- confusion
def confusion(labels, preds, n_classes):
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (labels, preds), 1)
    return cm / np.maximum(cm.sum(1, keepdims=True), 1)


def top_confused_pairs(cm, class_names, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:k], off.shape))[0]
    return [(class_names[i], class_names[j], float(off[i, j])) for i, j in idx if off[i, j] > 0]


def plot_confusion_top(cm, class_names, top_n=20, out=None):
    """Normalized confusion matrix restricted to the top_n classes with the highest error rate."""
    import seaborn as sns
    top_n = min(top_n, len(class_names))
    err = 1 - np.diag(cm)
    idx = np.sort(np.argsort(err)[-top_n:])
    sub = cm[np.ix_(idx, idx)]
    names = [class_names[i] for i in idx]
    size = max(6, 0.6 * top_n)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    sns.heatmap(sub, xticklabels=names, yticklabels=names, annot=top_n <= 25, fmt='.2f',
                cmap='Blues', vmin=0, vmax=1, ax=ax, annot_kws={'size': 7})
    ax.set(xlabel='predicted', ylabel='true', title=f'Confusion matrix ({top_n} most-confused classes)')
    fig.tight_layout()
    _save(fig, out)
    return fig


# ----------------------------------------------------------------------------- embeddings
def plot_tsne(X2d, labels, class_names, title, ax):
    n = len(class_names)
    cmap = plt.cm.tab20 if n <= 20 else plt.cm.gist_ncar
    colors = cmap(np.linspace(0, 1, max(n, 2)))
    for c in range(n):
        m = labels == c
        if m.any():
            ax.scatter(X2d[m, 0], X2d[m, 1], s=10, alpha=0.75, color=colors[c], label=class_names[c])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_distance_hist(pos, neg, title, ax, margin=None, bins=40):
    ax.hist(pos, bins=bins, range=(0, 2), alpha=0.6, density=True, label='same class')
    ax.hist(neg, bins=bins, range=(0, 2), alpha=0.6, density=True, label='different class')
    if margin is not None and margin <= 2:
        ax.axvline(margin, color='k', ls='--', lw=1, label=f'margin {margin}')
    ax.set(xlabel='embedding distance', title=title)
    ax.legend(fontsize=7)


# ----------------------------------------------------------------------------- per-class
def plot_per_class_gain(acc_a, acc_b, label_a, label_b, top=5, out=None):
    """acc_a / acc_b: dict class -> accuracy. Bars = acc_b - acc_a, largest gains highlighted."""
    names = sorted(set(acc_a) & set(acc_b))
    gain = pd.Series({n: acc_b[n] - acc_a[n] for n in names}).sort_values()
    fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(names) + 4), 4.5))
    colors = ['tab:green' if n in gain.index[-top:] else ('tab:red' if g < 0 else 'tab:gray')
              for n, g in gain.items()]
    ax.bar(range(len(gain)), gain.values, color=colors)
    ax.set_xticks(range(len(gain)))
    ax.set_xticklabels(gain.index, rotation=90, fontsize=7 if len(names) > 30 else 9)
    ax.axhline(0, color='k', lw=0.8)
    ax.set(ylabel=f'accuracy({label_b}) - accuracy({label_a})',
           title=f'Per-class gain of {label_b} over {label_a} (green = top {top})')
    fig.tight_layout()
    _save(fig, out)
    return fig, gain


# ----------------------------------------------------------------------------- Grad-CAM
class VideoGradCAM:
    """Grad-CAM for 3D CNNs: weight each channel of `layer`'s output by its mean gradient
    (over time, height and width), sum, ReLU, upsample to the clip size, normalize per frame.

    Uses a tensor hook on the layer output (robust to in-place ops) and makes the first input
    require grad, so it also works when the backbone is frozen.
    """

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.acts = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, 'grads', g.detach()))

    def __call__(self, inputs, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        inputs = [x.clone().requires_grad_(i == 0) for i, x in enumerate(inputs)]
        with torch.enable_grad():
            logits = self.model(*inputs)
            if class_idx is None:
                class_idx = int(logits.argmax(1)[0])
            logits[0, class_idx].backward()
        weights = self.grads.float().mean(dim=(2, 3, 4), keepdim=True)          # (1, C, 1, 1, 1)
        cam = torch.relu((weights * self.acts.float()).sum(dim=1, keepdim=True))  # (1, 1, t, h, w)
        t, h, w = inputs[0].shape[2:]
        cam = F.interpolate(cam, size=(t, h, w), mode='trilinear', align_corners=False)[0, 0]
        lo = cam.flatten(1).min(1).values[:, None, None]
        hi = cam.flatten(1).max(1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)
        prob = float(logits.detach().softmax(1)[0, class_idx])
        return cam.cpu().numpy(), class_idx, prob, tuple(self.acts.shape[2:])

    def remove(self):
        self.handle.remove()


def overlay_cam(frames, cam, alpha=0.45):
    """frames (T,H,W,3) float [0,1], cam (T,H,W) [0,1] -> (T,H,W,3) uint8 heat-map overlay."""
    heat = plt.cm.jet(cam)[..., :3]
    return (np.clip((1 - alpha) * frames + alpha * heat, 0, 1) * 255).astype(np.uint8)


def save_gif(frames_uint8, path, fps=4):
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, list(frames_uint8), duration=1.0 / fps, loop=0)
