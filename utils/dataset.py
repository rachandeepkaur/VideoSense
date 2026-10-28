"""VideoSense datasets: cached-frame loading, segment sampling, clip-consistent augmentation."""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NORM_STATS = {
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),           # C3D from scratch
    'kinetics': ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),  # torchvision video models
}


def decode_frames(data, offsets, which):
    """Decode selected JPEG frames from packed bytes -> (n, H, W, 3) uint8 RGB."""
    return np.stack([cv2.cvtColor(cv2.imdecode(data[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR),
                                  cv2.COLOR_BGR2RGB) for i in which])


def load_cached_frames(path, which=None):
    """Decode frames of a cached clip (all frames if which is None)."""
    z = np.load(path)
    data, off = z['data'], z['offsets']
    return decode_frames(data, off, range(len(off) - 1) if which is None else which)


def segment_indices(k, t, train):
    """Pick t of k stored frames: one per equal-width segment.

    train=True  -> random frame inside each segment (temporal augmentation)
    train=False -> fixed frame per segment (deterministic evaluation)
    """
    edges = np.linspace(0, k, t + 1)
    lo = np.floor(edges[:-1]).astype(int)
    hi = np.maximum(np.floor(edges[1:]).astype(int), lo + 1)
    if train:
        return [random.randrange(a, b) for a, b in zip(lo, hi)]
    return ((lo + hi - 1) // 2).tolist()


class ClipTransform:
    """Spatial crop + optional horizontal flip, identical for every frame of a clip."""

    def __init__(self, crop=112, train=False, hflip_p=0.5):
        self.crop, self.train, self.hflip_p = crop, train, hflip_p

    def sample_params(self, h, w):
        c = self.crop
        if self.train:
            return {'y': random.randint(0, h - c), 'x': random.randint(0, w - c),
                    'flip': random.random() < self.hflip_p}
        return {'y': (h - c) // 2, 'x': (w - c) // 2, 'flip': False}

    def __call__(self, frames, params=None):
        """frames: (T, H, W, C) array. Returns (cropped frames, params used)."""
        params = params or self.sample_params(*frames.shape[1:3])
        y, x, c = params['y'], params['x'], self.crop
        out = frames[:, y:y + c, x:x + c]
        if params['flip']:
            out = out[:, :, ::-1]
        return np.ascontiguousarray(out), params


def to_normalized_tensor(frames, norm='imagenet'):
    """(T, H, W, 3) uint8 -> (3, T, H, W) float32, channel-normalized."""
    mean, std = (torch.tensor(v).view(3, 1, 1, 1) for v in NORM_STATS[norm])
    x = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(255.0)
    return (x - mean) / std


def denormalize(clip, norm='imagenet'):
    """(3, T, H, W) normalized tensor -> (T, H, W, 3) float array in [0, 1] for plotting."""
    mean, std = (torch.tensor(v).view(3, 1, 1, 1) for v in NORM_STATS[norm])
    return (clip * std + mean).clamp(0, 1).permute(1, 2, 3, 0).numpy()


class UCF101Clips(Dataset):
    """RGB clips from the frame cache.

    df              : manifest rows with columns rel_path, label
    cache_dir       : folder with <Class>/<video>.npz files
    temporal_jitter : random frame per segment (defaults to train)
    Returns   : (clip (3, T, crop, crop) float32, label int)
    """

    def __init__(self, df, cache_dir, num_frames=16, train=False, crop=112, norm='imagenet',
                 temporal_jitter=None):
        self.df = df.reset_index(drop=True)
        self.cache_dir, self.num_frames, self.train, self.norm = cache_dir, num_frames, train, norm
        # temporal_jitter defaults to `train`; set it alone (train=False) for temporal test-time augmentation
        self.temporal_jitter = train if temporal_jitter is None else temporal_jitter
        self.transform = ClipTransform(crop=crop, train=train)
        self.labels = self.df.label.astype(int).tolist()
        self.samples = list(zip(self.df.rel_path, self.labels))      # assignment-compatible
        self.class_names = (self.df.drop_duplicates('label').sort_values('label').class_name.tolist())

    def __len__(self):
        return len(self.df)

    def path(self, i):
        return os.path.join(self.cache_dir, self.df.rel_path[i][:-4] + '.npz')

    def get(self, i):
        """Full output: clip, label, and the sampling/transform parameters used."""
        z = np.load(self.path(i))                      # one file read per clip
        data, off = z['data'], z['offsets']
        which = segment_indices(len(off) - 1, self.num_frames, self.temporal_jitter)
        frames, params = self.transform(decode_frames(data, off, which))
        params['frame_idx'] = which
        return to_normalized_tensor(frames, self.norm), self.labels[i], params

    def __getitem__(self, i):
        clip, label, _ = self.get(i)
        return clip, label
