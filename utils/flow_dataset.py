"""Flow and RGB+flow datasets that share frame sampling, crop and flip."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset import ClipTransform, segment_indices
from utils.optical_flow import decode_flow


class FlowClips(Dataset):
    """Optical-flow clips from the flow cache -> ((2, T, crop, crop) float32, label)."""

    def __init__(self, df, flow_dir, num_frames=16, train=False, crop=112, camera_comp=False,
                 temporal_jitter=None):
        self.df = df.reset_index(drop=True)
        self.flow_dir, self.num_frames, self.train = flow_dir, num_frames, train
        self.temporal_jitter = train if temporal_jitter is None else temporal_jitter
        self.camera_comp = camera_comp
        self.transform = ClipTransform(crop=crop, train=train)
        self.labels = self.df.label.astype(int).tolist()
        self.samples = list(zip(self.df.rel_path, self.labels))

    def __len__(self):
        return len(self.df)

    def path(self, i):
        return os.path.join(self.flow_dir, self.df.rel_path[i][:-4] + '.npz')

    def get(self, i, params=None):
        """params (from the RGB dataset) forces the same frames/crop/flip; None samples new ones."""
        z = np.load(self.path(i))
        data, off = z['data'], z['offsets']
        k = (len(off) - 1) // 2
        which = params['frame_idx'] if params else segment_indices(k, self.num_frames, self.temporal_jitter)
        assert max(which) < k, 'RGB and flow caches have different K'
        flow = decode_flow(data, off, which)                       # (T, H, W, 2) in [-1, 1]
        if self.camera_comp:
            flow = flow - np.median(flow, axis=(1, 2), keepdims=True)
        flow, p = self.transform(flow, params)
        if p['flip']:
            flow[..., 0] *= -1                                     # mirrored image: right <-> left
        p['frame_idx'] = which
        return torch.from_numpy(flow).permute(3, 0, 1, 2).float(), self.labels[i], p

    def __getitem__(self, i):
        flow, label, _ = self.get(i)
        return flow, label


class MultimodalClips(Dataset):
    """(rgb (3,T,H,W), flow (2,T,H,W), label) with one shared set of augmentation parameters."""

    def __init__(self, rgb_ds, flow_ds):
        assert list(rgb_ds.df.rel_path) == list(flow_ds.df.rel_path), 'datasets not aligned'
        self.rgb, self.flow = rgb_ds, flow_ds
        self.samples, self.labels = rgb_ds.samples, rgb_ds.labels

    def __len__(self):
        return len(self.rgb)

    def get(self, i):
        rgb, label, params = self.rgb.get(i)
        flow, _, _ = self.flow.get(i, params=params)
        return rgb, flow, label, params

    def __getitem__(self, i):
        rgb, flow, label, _ = self.get(i)
        return rgb, flow, label
