"""Shared infrastructure for all VideoSense training / evaluation scripts.

Data      : load splits + preprocessing config, restrict classes, build RGB / flow / RGB+flow datasets
Training  : one generic epoch loop (AMP, grad clipping), fit loop with val-based checkpointing,
            resume after a Colab disconnect, CSV history, early stopping
Testing   : top-1 / top-5 / macro-F1 / per-class accuracy on the untouched test split, saved to disk
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from utils.dataset import UCF101Clips
from utils.flow_dataset import FlowClips, MultimodalClips

COLAB_DATA_ROOT = '/content/videosense/data'
COLAB_META_DIR = '/content/drive/MyDrive/videosense/metadata'
COLAB_RUNS_DIR = '/content/drive/MyDrive/videosense/runs'


# ----------------------------------------------------------------------------- arguments
def add_common_args(parser):
    g = parser.add_argument_group('data / run')
    g.add_argument('--data-root', default=COLAB_DATA_ROOT, help='folder holding the frame/flow caches')
    g.add_argument('--meta-dir', default=COLAB_META_DIR, help='folder with splits_*.csv and preprocess_config.json')
    g.add_argument('--runs-dir', default=COLAB_RUNS_DIR, help='checkpoints, histories and metrics go here')
    g.add_argument('--subset', choices=['mini', 'full'], default='mini')
    g.add_argument('--run-name', required=True)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--num-workers', type=int, default=2)
    g.add_argument('--no-amp', action='store_true', help='disable mixed precision (on by default on GPU)')
    g.add_argument('--resume', action='store_true', help='continue from <run>/last.pt after a disconnect')
    g.add_argument('--trainval', action='store_true',
                   help='final run: train on train+val for a fixed number of epochs (no model selection)')
    g.add_argument('--export', default=None, help='also save the best state_dict here (Section 11.2 names)')
    g.add_argument('--max-batches', type=int, default=None, help='smoke test: limit batches per epoch')
    g.add_argument('--no-test', action='store_true', help='skip the final test-set evaluation')
    g.add_argument('--keep-last', action='store_true',
                   help='keep last.pt (optimizer state, ~3x model size) after a run finishes; by default it '
                        'is deleted to save Drive space (it is only needed to resume an interrupted run)')
    o = parser.add_argument_group('optimization')
    o.add_argument('--epochs', type=int, default=30)
    o.add_argument('--batch-size', type=int, default=8)
    o.add_argument('--lr', type=float, default=1e-3)
    o.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adam')
    o.add_argument('--weight-decay', type=float, default=1e-4)
    o.add_argument('--warmup-epochs', type=int, default=0)
    o.add_argument('--grad-clip', type=float, default=None)
    o.add_argument('--patience', type=int, default=None, help='early stopping (epochs without val improvement)')
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True       # fixed input size -> let cuDNN pick fastest kernels
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ----------------------------------------------------------------------------- data
def load_preprocess_config(meta_dir):
    with open(os.path.join(meta_dir, 'preprocess_config.json')) as f:
        return json.load(f)


def load_splits(meta_dir, subset):
    return pd.read_csv(os.path.join(meta_dir, f'splits_{subset}.csv'))


def restrict_classes(df, classes):
    """Keep only `classes` and relabel them 0..C-1 in sorted order."""
    classes = sorted(classes)
    mapping = {c: i for i, c in enumerate(classes)}
    out = df[df.class_name.isin(classes)].copy()
    out['label'] = out.class_name.map(mapping).astype(int)
    return out.reset_index(drop=True)


def fewshot_class_split(class_names, n_novel, seed=0):
    """Deterministic base / novel class split for few-shot experiments (same seed -> same split
    in train_3dcnn.py --exclude-novel and train_siamese.py)."""
    names = sorted(class_names)
    novel = sorted(random.Random(seed).sample(names, n_novel))
    return [c for c in names if c not in novel], novel


def make_data_spec(subset, classes, modality='rgb', norm='imagenet', camera_comp=False):
    """Everything needed to rebuild the exact datasets later (stored inside every checkpoint)."""
    return dict(subset=subset, classes=sorted(classes), modality=modality, norm=norm,
                camera_comp=camera_comp)


def phase_frames(spec, meta_dir, trainval=False):
    df = restrict_classes(load_splits(meta_dir, spec['subset']), spec['classes'])
    frames = {p: df[df.phase == p].reset_index(drop=True) for p in ('train', 'val', 'test')}
    if trainval:
        frames['train'] = pd.concat([frames['train'], frames['val']], ignore_index=True)
    return frames


def build_dataset(spec, df, data_root, cfg, train, temporal_jitter=None):
    rgb_dir = os.path.join(data_root, cfg['CACHE_NAME'])
    flow_dir = os.path.join(data_root, cfg['FLOW_NAME'])
    for d, needed in ((rgb_dir, spec['modality'] != 'flow'), (flow_dir, spec['modality'] != 'rgb')):
        if needed and not os.path.isdir(d):
            raise FileNotFoundError(f'{d} not found: restore the caches from Drive first '
                                    '(cell "Restore caches" in the experiments notebook).')
    kw = dict(num_frames=cfg['T'], train=train, crop=cfg['CROP'], temporal_jitter=temporal_jitter)
    if spec['modality'] == 'rgb':
        return UCF101Clips(df, rgb_dir, norm=spec['norm'], **kw)
    flow = FlowClips(df, flow_dir, camera_comp=spec.get('camera_comp', False), **kw)
    if spec['modality'] == 'flow':
        return flow
    return MultimodalClips(UCF101Clips(df, rgb_dir, norm=spec['norm'], **kw), flow)


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False, batch_sampler=None):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kw)


def build_loaders(spec, args, cfg):
    frames = phase_frames(spec, args.meta_dir, args.trainval)
    loaders = {'train': make_loader(build_dataset(spec, frames['train'], args.data_root, cfg, True),
                                    args.batch_size, True, args.num_workers, drop_last=True)}
    if not args.trainval:
        loaders['val'] = make_loader(build_dataset(spec, frames['val'], args.data_root, cfg, False),
                                     args.batch_size, False, args.num_workers)
    loaders['test'] = make_loader(build_dataset(spec, frames['test'], args.data_root, cfg, False),
                                  args.batch_size, False, args.num_workers)
    return loaders, frames


# ----------------------------------------------------------------------------- optimization
def param_groups(model, head_prefixes=(), backbone_lr_mult=1.0, base_lr=1e-3, exclude=()):
    """Two learning rates: pretrained backbone (base_lr * mult) and new head (base_lr).
    Parameters whose name starts with an `exclude` prefix are left out (caller adds them)."""
    head, body = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and not any(name.startswith(e) for e in exclude):
            (head if any(name.startswith(h) for h in head_prefixes) else body).append(p)
    groups = []
    if body:
        groups.append({'params': body, 'lr': base_lr * backbone_lr_mult})
    if head:
        groups.append({'params': head, 'lr': base_lr})
    return groups


def make_optimizer(groups, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    return cls(groups, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, epochs, warmup_epochs=0):
    """Linear warmup, then cosine decay to ~0 (stepped once per epoch)."""
    def factor(e):
        if e < warmup_epochs:
            return (e + 1) / (warmup_epochs + 1)
        progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_scaler(device, amp):
    enabled = amp and device.type == 'cuda'
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):                       # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------- epoch loop
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=True,
              max_batches=None, grad_clip=None, collect=False):
    """One pass over `loader`. Batches are (input_1, ..., input_n, labels); the model is called
    as model(input_1, ..., input_n), so the same loop serves RGB, flow and two-stream models.
    Returns loss, top-1, top-5 (and logits/labels if collect=True)."""
    train = optimizer is not None
    model.train(train)
    use_amp = amp and device.type == 'cuda'
    tot_loss, n, c1, c5 = 0.0, 0, 0, 0
    logits_all, labels_all, alpha_all = [], [], []
    with torch.set_grad_enabled(train):
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            *inputs, labels = batch
            inputs = [x.to(device, non_blocking=True) for x in inputs]
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(*inputs)
            loss = criterion(logits.float(), labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            bs = labels.size(0)
            tot_loss += loss.item() * bs
            n += bs
            topk = logits.topk(min(5, logits.size(1)), dim=1).indices
            c1 += (topk[:, 0] == labels).sum().item()
            c5 += (topk == labels[:, None]).any(dim=1).sum().item()
            if collect:
                logits_all.append(logits.float().cpu())
                labels_all.append(labels.cpu())
                if getattr(model, 'last_alpha', None) is not None:
                    alpha_all.append(model.last_alpha.cpu())
    res = dict(loss=tot_loss / max(n, 1), top1=c1 / max(n, 1), top5=c5 / max(n, 1))
    if collect:
        res['logits'], res['labels'] = torch.cat(logits_all), torch.cat(labels_all)
        if alpha_all:
            res['alpha'] = torch.cat(alpha_all)
    return res


def evaluate(model, loader, device, amp=True):
    """Exercise 6: top-1 and top-5 accuracy on a loader."""
    r = run_epoch(model, loader, device, nn.CrossEntropyLoss(), amp=amp)
    return r['top1'], r['top5']


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, meta, **extra):
    state = {'model_state': model.state_dict(), 'meta': meta}
    for k, v in extra.items():
        state[k] = v.state_dict() if hasattr(v, 'state_dict') else v
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)                                    # never leaves a half-written checkpoint


def load_run(run_dir, device, which='best'):
    """Rebuild a model from <run_dir>/<which>.pt using the metadata stored inside it."""
    from models import build_model
    ck = torch.load(os.path.join(run_dir, f'{which}.pt'), map_location=device, weights_only=False)
    meta = ck['meta']
    model = build_model(meta['model_type'], for_loading=True, **meta['model_kwargs'])
    model.load_state_dict(ck['model_state'])
    return model.to(device).eval(), meta


def write_history(history, run_dir):
    pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)


def fit_classifier(model, loaders, criterion, optimizer, scheduler, args, run_dir, meta, device,
                   epoch_extra=None):
    """Train with per-epoch validation; keep the checkpoint with the best val top-1 (best.pt) and
    the latest full state for resuming (last.pt). With --trainval the last epoch is kept."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    amp = not args.no_amp
    scaler = make_scaler(device, amp)
    best_path, last_path = os.path.join(run_dir, 'best.pt'), os.path.join(run_dir, 'last.pt')
    start, best, history, bad = 0, -1.0, [], 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        start, best, history, bad = ck['epoch'] + 1, ck['best'], ck['history'], ck.get('bad', 0)
        print(f'Resumed from epoch {start}')
    use_val = 'val' in loaders
    for epoch in range(start, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[-1]['lr']
        tr = run_epoch(model, loaders['train'], device, criterion, optimizer, scaler, amp,
                       args.max_batches, args.grad_clip)
        va = run_epoch(model, loaders['val'], device, criterion, amp=amp,
                       max_batches=args.max_batches) if use_val else {}
        scheduler.step()
        row = {'epoch': epoch + 1, 'lr': lr, **{f'train_{k}': v for k, v in tr.items()},
               **{f'val_{k}': v for k, v in va.items()}, 'time_s': round(time.time() - t0, 1)}
        if epoch_extra:
            row.update(epoch_extra(model))
        history.append(row)
        write_history(history, run_dir)
        score = va['top1'] if use_val else float(epoch)
        if score > best:
            best, bad = score, 0
            save_checkpoint(best_path, model, dict(meta, epoch=epoch + 1))
        else:
            bad += 1
        save_checkpoint(last_path, model, meta, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, best=best, history=history, bad=bad)
        msg = (f'Epoch {epoch + 1:3d}/{args.epochs} | train loss {tr["loss"]:.4f} '
               f'top1 {tr["top1"]:.3f}')
        if use_val:
            msg += f' | val loss {va["loss"]:.4f} top1 {va["top1"]:.3f} top5 {va["top5"]:.3f}'
        print(msg + f' | lr {lr:.2e} | {row["time_s"]}s' + (' *' if bad == 0 else ''), flush=True)
        if args.patience and use_val and bad >= args.patience:
            print(f'Early stopping: no val improvement for {args.patience} epochs')
            break
    cleanup_last(last_path, args)
    return best_path


def cleanup_last(last_path, args):
    if not args.keep_last and os.path.exists(last_path):
        os.remove(last_path)


# ----------------------------------------------------------------------------- test
def final_test(run_dir, test_loader, device, amp=True, max_batches=None):
    """Evaluate best.pt ONCE on the test split; save metrics + raw outputs for later analysis."""
    model, meta = load_run(run_dir, device)
    res = run_epoch(model, test_loader, device, nn.CrossEntropyLoss(), amp=amp,
                    max_batches=max_batches, collect=True)
    logits, labels = res['logits'], res['labels']
    preds = logits.argmax(1)
    names = meta['data']['classes']
    per_class = {names[c]: float((preds[labels == c] == c).float().mean()) for c in range(len(names))
                 if (labels == c).any()}
    metrics = dict(top1=res['top1'], top5=res['top5'], loss=res['loss'],
                   macro_f1=float(f1_score(labels, preds, average='macro')),
                   n_clips=int(len(labels)), best_epoch=meta.get('epoch'), per_class_acc=per_class)
    if 'alpha' in res:
        metrics['mean_alpha_rgb'] = float(res['alpha'].mean())
    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    rel_paths = [s[0] for s in test_loader.dataset.samples][:len(labels)]
    np.savez(os.path.join(run_dir, 'test_outputs.npz'), logits=logits.numpy(), labels=labels.numpy(),
             rel_paths=np.array(rel_paths), **({'alpha': res['alpha'].numpy()} if 'alpha' in res else {}))
    print(f'TEST | top1 {metrics["top1"]:.4f} | top5 {metrics["top5"]:.4f} | macro-F1 {metrics["macro_f1"]:.4f}')
    return metrics


def export_state_dict(run_dir, path, device='cpu'):
    """Section 11.2 format: torch.save(model.state_dict(), path)."""
    model, _ = load_run(run_dir, torch.device(device))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Exported state_dict -> {path}')


def str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y')


def parser(description):
    return add_common_args(argparse.ArgumentParser(description=description))
