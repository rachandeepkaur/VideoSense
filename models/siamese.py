"""Siamese network for video similarity (Part 3).

VideoSiamese        : twin encoder with shared weights -> L2-normalized embedding -> distance
ContrastiveLoss     : Hadsell et al. (2006); label 0 = same class, 1 = different class
TripletLoss         : max(0, d(a,p) - d(a,n) + margin)
batch_hard_triplet_loss : hardest positive / negative inside a P x K batch (Hermans et al., 2017)
PairDataset / TripletDataset / FixedPairs / PKBatchSampler : data samplers
few_shot_eval       : N-way K-shot episodes on precomputed embeddings
"""
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from models.cnn3d import VideoEncoder


class VideoSiamese(nn.Module):
    """Twin encoders with tied weights. In code this is ONE encoder applied to both clips:
    weight sharing is what makes the two embeddings comparable."""

    def __init__(self, embed_dim=512, dropout=0.5, backbone='c3d', pretrained=False, batch_norm=False):
        super().__init__()
        self.encoder = VideoEncoder(backbone, 3, pretrained=pretrained, batch_norm=batch_norm)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, embed_dim))
        self.embed_dim = embed_dim

    def encode(self, x):
        """L2-normalized embedding (on the unit sphere, so all distances lie in [0, 2])."""
        emb = self.projector(self.encoder(x)).float()
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1, x2):
        """Euclidean distance between the embeddings of x1 and x2 (small = similar)."""
        return pair_distance(self.encode(x1), self.encode(x2))

    def similarity(self, x1, x2):
        return F.cosine_similarity(self.encode(x1), self.encode(x2))

    def get_embedding(self, x):
        return self.encode(x)

    @property
    def cam_layer(self):
        return self.encoder.cam_layer


def pair_distance(z1, z2):
    """Euclidean distance; the epsilon keeps the sqrt gradient finite when z1 == z2."""
    return torch.sqrt(((z1 - z2) ** 2).sum(dim=1) + 1e-8)


class ContrastiveLoss(nn.Module):
    """L = (1-y) * 0.5 * d^2  +  y * 0.5 * max(0, margin - d)^2

    y=0 (same class)      : pulled together.
    y=1 (different class) : pushed apart until d >= margin, then ignored.

    With L2-normalized embeddings d <= 2 always: margin=2 asks negatives to be on opposite sides
    of the sphere, and margin>2 can never be satisfied, so every negative pair keeps pushing.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        similar = (1 - label) * 0.5 * dist.pow(2)
        dissimilar = label * 0.5 * torch.clamp(self.margin - dist, min=0).pow(2)
        return (similar + dissimilar).mean()


class TripletLoss(nn.Module):
    """L = max(0, d(a, p) - d(a, n) + margin): the negative must be farther than the positive
    by at least `margin`. Only relative distances matter, unlike contrastive loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, za, zp, zn):
        return F.relu(pair_distance(za, zp) - pair_distance(za, zn) + self.margin).mean()


def batch_hard_triplet_loss(z, labels, margin=0.3):
    """For each anchor in the batch: hardest (farthest) positive and hardest (closest) negative.
    Needs several clips per class in the batch -> use PKBatchSampler."""
    d = torch.cdist(z, z)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    hardest_pos = (d * (same & ~eye)).max(dim=1).values
    hardest_neg = d.masked_fill(same, float('inf')).min(dim=1).values
    valid = torch.isfinite(hardest_neg) & (same & ~eye).any(dim=1)
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean() if valid.any() else z.sum() * 0


# ----------------------------------------------------------------------------- datasets
def _class_index(labels):
    idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        idx[int(lbl)].append(i)
    return idx


class PairDataset(Dataset):
    """Random clip pairs: label 0 = same class (positive), 1 = different classes (negative).

    pos_fraction=0.5 gives the balanced pairs of the assignment; change it to study Q19.
    """

    def __init__(self, base_dataset, pairs_per_epoch=10000, pos_fraction=0.5):
        self.ds, self.pairs_per_epoch, self.pos_fraction = base_dataset, pairs_per_epoch, pos_fraction
        self.class_idx = _class_index(base_dataset.labels)
        self.classes = [c for c, v in self.class_idx.items() if len(v) >= 2]

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, _):
        if random.random() < self.pos_fraction:
            cls = random.choice(self.classes)
            i1, i2 = random.sample(self.class_idx[cls], 2)
            label = 0
        else:
            c1, c2 = random.sample(self.classes, 2)
            i1, i2 = random.choice(self.class_idx[c1]), random.choice(self.class_idx[c2])
            label = 1
        return self.ds[i1][0], self.ds[i2][0], torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    """Random (anchor, positive, negative) triplets."""

    def __init__(self, base_dataset, triplets_per_epoch=10000):
        self.ds, self.n = base_dataset, triplets_per_epoch
        self.class_idx = _class_index(base_dataset.labels)
        self.classes = [c for c, v in self.class_idx.items() if len(v) >= 2]

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        c_pos, c_neg = random.sample(self.classes, 2)
        a, p = random.sample(self.class_idx[c_pos], 2)
        n = random.choice(self.class_idx[c_neg])
        return self.ds[a][0], self.ds[p][0], self.ds[n][0]


class FixedPairs(Dataset):
    """A fixed, seeded list of balanced pairs, so validation numbers are comparable across epochs."""

    def __init__(self, base_dataset, n_pairs=400, seed=0):
        rng = random.Random(seed)
        idx = _class_index(base_dataset.labels)
        classes = [c for c, v in idx.items() if len(v) >= 2]
        self.ds, self.pairs = base_dataset, []
        for k in range(n_pairs):
            if k % 2 == 0:
                c = rng.choice(classes)
                self.pairs.append((*rng.sample(idx[c], 2), 0))
            else:
                c1, c2 = rng.sample(classes, 2)
                self.pairs.append((rng.choice(idx[c1]), rng.choice(idx[c2]), 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, y = self.pairs[k]
        return self.ds[i][0], self.ds[j][0], torch.tensor(y, dtype=torch.float32)


class PKBatchSampler(Sampler):
    """Batches of P classes x K clips: every batch contains positives AND negatives for every
    anchor, which batch-hard mining needs. One clip is encoded once but used in many pairs."""

    def __init__(self, labels, p=4, k=4, batches_per_epoch=500):
        self.class_idx = _class_index(labels)
        self.classes = [c for c, v in self.class_idx.items() if len(v) >= 2]
        self.p, self.k, self.n = min(p, len(self.classes)), k, batches_per_epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            batch = []
            for c in random.sample(self.classes, self.p):
                pool = self.class_idx[c]
                batch += random.sample(pool, self.k) if len(pool) >= self.k else random.choices(pool, k=self.k)
            yield batch


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=16, num_workers=2):
    """Encode every clip ONCE. Episodes then only index into this matrix, which is ~100x cheaper
    than re-encoding clips inside every episode (as the assignment template does)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs, labels = [], []
    for batch in loader:
        embs.append(model.encode(batch[0].to(device)).float().cpu())
        labels.append(batch[-1])
    return torch.cat(embs), torch.cat(labels)


def few_shot_eval(embs, labels, groups=None, n_way=5, k_shot=1, n_query=15, n_episodes=200,
                  method='nn', seed=0):
    """N-way K-shot accuracy over random episodes.

    method='nn'    : 1-nearest-neighbour to any support clip (the assignment's protocol)
    method='proto' : nearest class mean of the support clips (Prototypical-Network style inference)
    groups         : if given, queries never share a UCF101 group with that class's support clips
                     (otherwise a query can be a near-duplicate cut of the same video -> inflated).
    Returns mean, std, 95% confidence interval half-width.
    """
    rng = random.Random(seed)
    by_class = _class_index(labels.tolist())
    eligible = [c for c, v in by_class.items() if len(v) > k_shot]
    if len(eligible) < n_way:
        raise ValueError(f'only {len(eligible)} classes available for {n_way}-way episodes')
    accs = []
    for _ in range(n_episodes):
        sup, sup_lbl, qry, qry_lbl = [], [], [], []
        for li, c in enumerate(rng.sample(eligible, n_way)):
            idxs = by_class[c][:]
            rng.shuffle(idxs)
            s = idxs[:k_shot]
            pool = idxs[k_shot:]
            if groups is not None:
                used = {groups[i] for i in s}
                pool = [i for i in pool if groups[i] not in used] or pool
            q = pool[:n_query]
            sup += s
            sup_lbl += [li] * len(s)
            qry += q
            qry_lbl += [li] * len(q)
        S, Q = embs[sup], embs[qry]
        sup_lbl, qry_lbl = torch.tensor(sup_lbl), torch.tensor(qry_lbl)
        if method == 'proto':
            protos = torch.stack([S[sup_lbl == c].mean(0) for c in range(n_way)])
            pred = torch.cdist(Q, protos).argmin(1)
        else:
            pred = sup_lbl[torch.cdist(Q, S).argmin(1)]
        accs.append((pred == qry_lbl).float().mean().item())
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std()), float(1.96 * accs.std() / math.sqrt(len(accs)))
