"""Part 2: train a 3D CNN for action classification (Exercises 6-9).

Examples
--------
# C3D from scratch (lr defaults to 3e-4: see the note in main())
python train_3dcnn.py --run-name c3d_ls0.1 --model c3d --label-smoothing 0.1 --epochs 30

# Label-smoothing sweep (Exercise 8)
python train_3dcnn.py --run-name c3d_ls0.0 --model c3d --label-smoothing 0.0
python train_3dcnn.py --run-name c3d_ls0.2 --model c3d --label-smoothing 0.2

# Pretrained R(2+1)D-18, train only the classifier head for 10 epochs (Exercise 9)
python train_3dcnn.py --run-name r2p1d_head --model r2plus1d --finetune head --epochs 10 --lr 1e-3

# C3D on the few-shot BASE classes only (clean initialization for the Siamese network, Part 3)
python train_3dcnn.py --run-name c3d_base --model c3d --exclude-novel 5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn  # noqa: E402

from models import build_model  # noqa: E402
from utils import common as U  # noqa: E402


def main():
    p = U.parser(__doc__.split('\n')[0])
    # Adam at the assignment's lr=1e-3 makes C3D's loss spike (2.3 -> ~445) after the first step;
    # 3e-4 trains smoothly. Pass --lr 1e-3 to reproduce the comparison.
    p.set_defaults(lr=3e-4)
    m = p.add_argument_group('model')
    m.add_argument('--model', choices=['c3d', 'r2plus1d', 'r2plus1d_custom'], default='c3d')
    m.add_argument('--label-smoothing', type=float, default=0.1)
    m.add_argument('--dropout', type=float, default=0.5)
    m.add_argument('--bn', action='store_true', help='C3D: add BatchNorm after every conv')
    m.add_argument('--finetune', choices=['head', 'layer4', 'all'], default='head',
                   help='r2plus1d: which part to train (head = Exercise 9)')
    m.add_argument('--no-pretrained', action='store_true', help='r2plus1d: random init')
    m.add_argument('--backbone-lr-mult', type=float, default=1.0,
                   help='lr multiplier for non-head parameters (e.g. 0.1 when fine-tuning pretrained layers)')
    m.add_argument('--exclude-novel', type=int, default=0,
                   help='train only on base classes, holding out this many novel classes (Part 3 init)')
    m.add_argument('--fewshot-seed', type=int, default=0)
    args = p.parse_args()

    U.set_seed(args.seed)
    device = U.get_device()
    cfg = U.load_preprocess_config(args.meta_dir)
    classes = sorted(U.load_splits(args.meta_dir, args.subset).class_name.unique())
    if args.exclude_novel:
        classes, novel = U.fewshot_class_split(classes, args.exclude_novel, args.fewshot_seed)
        print(f'Training on {len(classes)} base classes; held-out novel classes: {novel}')

    norm = 'kinetics' if args.model == 'r2plus1d' else 'imagenet'
    spec = U.make_data_spec(args.subset, classes, 'rgb', norm)
    loaders, frames = U.build_loaders(spec, args, cfg)
    print({k: len(v) for k, v in frames.items()}, 'clips | device:', device)

    n = len(classes)
    if args.model == 'c3d':
        kwargs, head = dict(num_classes=n, dropout=args.dropout, batch_norm=args.bn), ('classifier.',)
    elif args.model == 'r2plus1d':
        pretrained = not args.no_pretrained
        finetune = args.finetune if pretrained else 'all'
        kwargs = dict(num_classes=n, pretrained=pretrained, freeze_backbone=finetune != 'all',
                      train_layer4=finetune == 'layer4')
        head = ('fc.',)
    else:
        kwargs, head = dict(num_classes=n, dropout=args.dropout), ('fc.',)

    model = build_model(args.model, **kwargs).to(device)
    total, trainable = U.count_params(model)
    print(f'{args.model}: {total / 1e6:.2f}M parameters, {trainable / 1e6:.2f}M trainable')

    optimizer = U.make_optimizer(U.param_groups(model, head, args.backbone_lr_mult, args.lr), args)
    scheduler = U.make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    meta = dict(model_type=args.model, model_kwargs=kwargs, data=spec, args=vars(args), part='part2',
                params_total=total, params_trainable=trainable)
    run_dir = os.path.join(args.runs_dir, args.run_name)

    U.fit_classifier(model, loaders, criterion, optimizer, scheduler, args, run_dir, meta, device)
    if not args.no_test:
        U.final_test(run_dir, loaders['test'], device, not args.no_amp, args.max_batches)
    if args.export:
        U.export_state_dict(run_dir, args.export)


if __name__ == '__main__':
    main()
