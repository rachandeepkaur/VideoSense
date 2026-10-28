"""Part 3: Siamese network for video similarity + few-shot evaluation (Exercises 14-17).

Classes are split into BASE (used for training) and NOVEL (never seen; used only for N-way K-shot
evaluation). Validation uses a fixed set of base-class pairs from the val split.

Examples
--------
# Contrastive loss, margin sweep (Exercises 14-16)
python train_siamese.py --run-name siam_m1 --loss contrastive --margin 1.0
python train_siamese.py --run-name siam_m2 --loss contrastive --margin 2.0
python train_siamese.py --run-name siam_m4 --loss contrastive --margin 4.0

# Triplet loss (Exercise 17) and the batch-hard variant
python train_siamese.py --run-name siam_triplet --loss triplet --margin 0.3
python train_siamese.py --run-name siam_bh --loss triplet_batchhard --margin 0.3

# Start from a C3D trained on base classes only (train_3dcnn.py --exclude-novel 5)
python train_siamese.py --run-name siam_m1_init --loss contrastive --margin 1.0 --init-from <runs>/c3d_base
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from models import build_model  # noqa: E402
from models.cnn3d import load_encoder_weights  # noqa: E402
from models.siamese import (ContrastiveLoss, FixedPairs, PairDataset, PKBatchSampler,  # noqa: E402
                            TripletDataset, TripletLoss, batch_hard_triplet_loss, embed_dataset,
                            few_shot_eval, pair_distance)
from utils import common as U  # noqa: E402

DEFAULT_MARGIN = {'contrastive': 1.0, 'triplet': 0.3, 'triplet_batchhard': 0.3}


def resolve_ckpt(path):
    return os.path.join(path, 'best.pt') if os.path.isdir(path) else path


def train_epoch(model, loader, args, device, scaler, optimizer, criterion):
    model.train()
    use_amp = not args.no_amp and device.type == 'cuda'
    total, n = 0.0, 0
    for b, batch in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        batch = [x.to(device, non_blocking=True) for x in batch]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if args.loss == 'contrastive':
                x1, x2, y = batch
                z = model.encode(torch.cat([x1, x2]))          # one pass: BatchNorm sees both halves
                loss = criterion(pair_distance(*z.chunk(2)), y)
            elif args.loss == 'triplet':
                za, zp, zn = model.encode(torch.cat(batch)).chunk(3)
                loss = criterion(za, zp, zn)
            else:
                x, y = batch
                loss = batch_hard_triplet_loss(model.encode(x), y, args.margin)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss.float()).backward()
        if args.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, args, device):
    """Distances for fixed same/different-class pairs -> AUC, separation, contrastive loss."""
    model.eval()
    dists, ys = [], []
    for b, (x1, x2, y) in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        dists.append(model(x1.to(device), x2.to(device)).float().cpu())
        ys.append(y)
    d, y = torch.cat(dists).numpy(), torch.cat(ys).numpy()
    pos, neg = d[y == 0], d[y == 1]
    auc = roc_auc_score(y, d) if 0 < y.sum() < len(y) else float('nan')
    sep = (neg.mean() - pos.mean()) / (np.sqrt(0.5 * (pos.var() + neg.var())) + 1e-8)
    closs = ContrastiveLoss(args.margin)(torch.tensor(d), torch.tensor(y)).item()
    return dict(val_auc=float(auc), val_separation=float(sep), val_pos_dist=float(pos.mean()),
                val_neg_dist=float(neg.mean()), val_contrastive_loss=closs), pos, neg


def main():
    p = U.parser(__doc__.split('\n')[0])
    p.set_defaults(lr=1e-4, epochs=20)
    m = p.add_argument_group('siamese')
    m.add_argument('--loss', choices=['contrastive', 'triplet', 'triplet_batchhard'], default='contrastive')
    m.add_argument('--margin', type=float, default=None, help='default: 1.0 contrastive, 0.3 triplet')
    m.add_argument('--embed-dim', type=int, default=512)
    m.add_argument('--dropout', type=float, default=0.5)
    m.add_argument('--backbone', choices=['c3d', 'r2plus1d'], default='c3d')
    m.add_argument('--pretrained', action='store_true', help='r2plus1d: Kinetics-400 weights')
    m.add_argument('--bn', action='store_true')
    m.add_argument('--init-from', default=None, help='run dir / checkpoint to copy encoder weights from')
    m.add_argument('--n-novel', type=int, default=5)
    m.add_argument('--fewshot-seed', type=int, default=0)
    m.add_argument('--pairs-per-epoch', type=int, default=2000,
                   help='compute budget; triplet modes encode the same number of clips per epoch')
    m.add_argument('--pos-fraction', type=float, default=0.5, help='share of positive pairs (Q19)')
    m.add_argument('--pk-p', type=int, default=4, help='batch-hard: classes per batch')
    m.add_argument('--pk-k', type=int, default=4, help='batch-hard: clips per class')
    m.add_argument('--val-pairs', type=int, default=400)
    m.add_argument('--episodes', type=int, default=200)
    m.add_argument('--n-way', type=int, default=5)
    m.add_argument('--n-query', type=int, default=15)
    args = p.parse_args()
    args.margin = DEFAULT_MARGIN[args.loss] if args.margin is None else args.margin

    U.set_seed(args.seed)
    device = U.get_device()
    cfg = U.load_preprocess_config(args.meta_dir)
    all_classes = sorted(U.load_splits(args.meta_dir, args.subset).class_name.unique())
    base, novel = U.fewshot_class_split(all_classes, args.n_novel, args.fewshot_seed)
    print(f'Base classes ({len(base)}): {base}\nNovel classes ({len(novel)}): {novel}')

    norm = 'kinetics' if args.backbone == 'r2plus1d' else 'imagenet'
    spec = U.make_data_spec(args.subset, base, 'rgb', norm)
    frames = U.phase_frames(spec, args.meta_dir, args.trainval)
    train_ds = U.build_dataset(spec, frames['train'], args.data_root, cfg, train=True)
    val_ds = U.build_dataset(spec, frames['val'], args.data_root, cfg, train=False)

    clips_budget = 2 * args.pairs_per_epoch
    if args.loss == 'contrastive':
        train_loader = U.make_loader(PairDataset(train_ds, args.pairs_per_epoch, args.pos_fraction),
                                     args.batch_size, True, args.num_workers, drop_last=True)
        criterion = ContrastiveLoss(args.margin)
    elif args.loss == 'triplet':
        train_loader = U.make_loader(TripletDataset(train_ds, clips_budget // 3),
                                     args.batch_size, True, args.num_workers, drop_last=True)
        criterion = TripletLoss(args.margin)
    else:
        sampler = PKBatchSampler(train_ds.labels, args.pk_p, args.pk_k,
                                 max(1, clips_budget // (args.pk_p * args.pk_k)))
        train_loader = U.make_loader(train_ds, None, False, args.num_workers, batch_sampler=sampler)
        criterion = None
    val_loader = U.make_loader(FixedPairs(val_ds, args.val_pairs, seed=0), args.batch_size, False,
                               args.num_workers)

    kwargs = dict(embed_dim=args.embed_dim, dropout=args.dropout, backbone=args.backbone,
                  pretrained=args.pretrained, batch_norm=args.bn)
    model = build_model('siamese', **kwargs)
    if args.init_from:
        state = torch.load(resolve_ckpt(args.init_from), map_location='cpu', weights_only=False)['model_state']
        load_encoder_weights(model.encoder, state)
        print('Encoder initialized from', args.init_from)
    model.to(device)

    optimizer = U.make_optimizer(U.param_groups(model, ('projector.',), 1.0, args.lr), args)
    scheduler = U.make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = U.make_scaler(device, not args.no_amp)
    run_dir = os.path.join(args.runs_dir, args.run_name)
    dist_dir = os.path.join(run_dir, 'distances')
    os.makedirs(dist_dir, exist_ok=True)
    meta = dict(model_type='siamese', model_kwargs=kwargs, data=spec, args=vars(args), part='part3',
                base_classes=base, novel_classes=novel)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    best_path, last_path = os.path.join(run_dir, 'best.pt'), os.path.join(run_dir, 'last.pt')
    start, best, history = 0, -1.0, []
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        start, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        print(f'Resumed from epoch {start}')

    for epoch in range(start, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[-1]['lr']
        tr_loss = train_epoch(model, train_loader, args, device, scaler, optimizer, criterion)
        val, pos, neg = validate(model, val_loader, args, device)
        scheduler.step()
        np.savez(os.path.join(dist_dir, f'epoch_{epoch + 1:03d}.npz'), pos=pos, neg=neg)
        row = dict(epoch=epoch + 1, lr=lr, train_loss=tr_loss, **val, time_s=round(time.time() - t0, 1))
        history.append(row)
        U.write_history(history, run_dir)
        improved = val['val_auc'] > best
        if improved:
            best = val['val_auc']
            U.save_checkpoint(best_path, model, dict(meta, epoch=epoch + 1))
        U.save_checkpoint(last_path, model, meta, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                          epoch=epoch, best=best, history=history)
        print(f'Epoch {epoch + 1:3d}/{args.epochs} | train loss {tr_loss:.4f} | val AUC {val["val_auc"]:.3f} '
              f'| d(pos) {val["val_pos_dist"]:.3f} d(neg) {val["val_neg_dist"]:.3f} '
              f'| sep {val["val_separation"]:.2f} | {row["time_s"]}s' + (' *' if improved else ''), flush=True)

    U.cleanup_last(last_path, args)
    if args.no_test:
        return
    # ---- few-shot evaluation on NOVEL classes (never seen in training), best checkpoint
    model, _ = U.load_run(run_dir, device)
    nspec = U.make_data_spec(args.subset, novel, 'rgb', norm)
    ndf = U.restrict_classes(U.load_splits(args.meta_dir, args.subset), novel)
    if args.max_batches:
        ndf = ndf.groupby('label').head(args.max_batches * 2).reset_index(drop=True)
    nds = U.build_dataset(nspec, ndf, args.data_root, cfg, train=False)
    embs, labels = embed_dataset(model, nds, device, args.batch_size, args.num_workers)
    groups = ndf.group.tolist()
    min_per_class = int(np.bincount(labels.numpy()).min())
    results = {'novel_classes': novel, 'n_way': min(args.n_way, len(novel))}
    for k in (1, 5):
        if min_per_class <= k:
            print(f'Skipping {k}-shot: a novel class has only {min_per_class} clips')
            continue
        n_query = max(1, min(args.n_query, min_per_class - k))
        for method in ('nn', 'proto'):
            mean, std, ci = few_shot_eval(embs, labels, groups, results['n_way'], k, n_query,
                                          args.episodes, method, seed=0)
            results[f'{results["n_way"]}way_{k}shot_{method}'] = dict(mean=mean, std=std, ci95=ci,
                                                                      n_query=n_query)
            print(f'NOVEL {results["n_way"]}-way {k}-shot ({method:5s}): {mean:.3f} +/- {std:.3f} '
                  f'(95% CI +/- {ci:.3f})')
    with open(os.path.join(run_dir, 'fewshot.json'), 'w') as f:
        json.dump(results, f, indent=2)
    if args.export:
        U.export_state_dict(run_dir, args.export)


if __name__ == '__main__':
    main()
