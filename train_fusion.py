"""Part 4: two-stream RGB + optical-flow models (Exercises 21-24).

All modes read the SAME (rgb, flow, label) batches with shared augmentation, so the ablation
differs only in what the model uses.

Examples
--------
# Ablation (Exercise 22)
python train_fusion.py --run-name fus_rgb    --mode rgb
python train_fusion.py --run-name fus_flow   --mode flow
python train_fusion.py --run-name fus_concat --mode concat          # also Exercise 21

# Attention-weighted fusion (Exercise 23): alpha logged every epoch and saved in test_metrics.json
python train_fusion.py --run-name fus_attn --mode attention

# Stronger: initialize each stream from its single-stream run, then train the fusion
python train_fusion.py --run-name fus_concat_init --mode concat \
       --init-rgb <runs>/fus_rgb --init-flow <runs>/fus_flow --backbone-lr-mult 0.1
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models import build_model  # noqa: E402
from models.cnn3d import load_encoder_weights  # noqa: E402
from models.fusion import MODES  # noqa: E402
from utils import common as U  # noqa: E402


def resolve_ckpt(path):
    return os.path.join(path, 'best.pt') if os.path.isdir(path) else path


def init_stream(model, stream, path, prefix=None):
    """Copy encoder (and, if shapes match, head) weights for one stream from another run."""
    state = torch.load(resolve_ckpt(path), map_location='cpu', weights_only=False)['model_state']
    encoder = getattr(model, f'{stream}_encoder')
    prefixes = [prefix] if prefix else [f'{stream}_encoder.', '', 'encoder.', 'rgb_encoder.', 'flow_encoder.']
    for pf in prefixes:
        try:
            load_encoder_weights(encoder, state, pf)
            break
        except KeyError:
            continue
    else:
        raise KeyError(f'no encoder weights in {path}')
    head = getattr(model, f'{stream}_head')
    head_state = {k[len(f'{stream}_head.'):]: v for k, v in state.items() if k.startswith(f'{stream}_head.')}
    if head_state and all(head.state_dict()[k].shape == v.shape for k, v in head_state.items()):
        head.load_state_dict(head_state)
        print(f'{stream} stream: encoder + head initialized from {path} (prefix "{pf}")')
    else:
        print(f'{stream} stream: encoder initialized from {path} (prefix "{pf}")')


def freeze(module):
    """Freeze weights AND keep BatchNorm statistics fixed (module stays in eval mode)."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    module.train = types.MethodType(lambda self, mode=True: self, module)


def main():
    p = U.parser(__doc__.split('\n')[0])
    p.set_defaults(lr=3e-4)          # each stream is a C3D: same first-step blow-up at 1e-3 (see train_3dcnn.py)
    m = p.add_argument_group('fusion')
    m.add_argument('--mode', choices=MODES, default='concat')
    m.add_argument('--backbone', choices=['c3d', 'r2plus1d'], default='c3d')
    m.add_argument('--pretrained', action='store_true', help='r2plus1d: Kinetics-400 weights for both streams')
    m.add_argument('--bn', action='store_true', help='C3D: BatchNorm after every conv')
    m.add_argument('--embed-dim', type=int, default=512)
    m.add_argument('--dropout', type=float, default=0.5)
    m.add_argument('--label-smoothing', type=float, default=0.1)
    m.add_argument('--camera-comp', action='store_true', help='subtract per-frame median flow (Step 4.3)')
    m.add_argument('--init-rgb', default=None, help='run dir / checkpoint for the RGB stream')
    m.add_argument('--init-flow', default=None, help='run dir / checkpoint for the flow stream')
    m.add_argument('--init-flow-prefix', default=None)
    m.add_argument('--freeze-encoders', action='store_true', help='train only heads + fusion')
    m.add_argument('--backbone-lr-mult', type=float, default=1.0)
    m.add_argument('--alpha-lr-mult', type=float, default=10.0,
                   help='attention mode: lr multiplier for the single scalar alpha (no weight decay)')
    args = p.parse_args()

    U.set_seed(args.seed)
    device = U.get_device()
    cfg = U.load_preprocess_config(args.meta_dir)
    classes = sorted(U.load_splits(args.meta_dir, args.subset).class_name.unique())
    norm = 'kinetics' if args.backbone == 'r2plus1d' else 'imagenet'
    spec = U.make_data_spec(args.subset, classes, 'multimodal', norm, args.camera_comp)
    loaders, frames = U.build_loaders(spec, args, cfg)
    print({k: len(v) for k, v in frames.items()}, 'clips | device:', device)

    kwargs = dict(num_classes=len(classes), mode=args.mode, backbone=args.backbone,
                  embed_dim=args.embed_dim, dropout=args.dropout, pretrained=args.pretrained,
                  batch_norm=args.bn)
    model = build_model('fusion', **kwargs)
    if args.init_rgb and model.use_rgb:
        init_stream(model, 'rgb', args.init_rgb)
    if args.init_flow and model.use_flow:
        init_stream(model, 'flow', args.init_flow, args.init_flow_prefix)
    if args.freeze_encoders:
        for s in ('rgb', 'flow'):
            if hasattr(model, f'{s}_encoder'):
                freeze(getattr(model, f'{s}_encoder'))
    model.to(device)
    total, trainable = U.count_params(model)
    print(f'fusion[{args.mode}]: {total / 1e6:.2f}M parameters, {trainable / 1e6:.2f}M trainable')

    head_prefixes = ('classifier.', 'rgb_cls.', 'flow_cls.', 'gate.', 'norm_', 'rgb_head.',
                     'flow_head.', 'rgb_tok.', 'flow_tok.', 'flow_to_rgb.', 'rgb_to_flow.', 'ln_')
    groups = U.param_groups(model, head_prefixes, args.backbone_lr_mult, args.lr, exclude=('alpha_logit',))
    if args.mode == 'attention':
        # Adam moves a scalar by ~lr per step, so alpha would barely leave 0.5 at the network's lr;
        # weight decay would also pull its logit back to 0 (= alpha 0.5). Own group: higher lr, no decay.
        groups.insert(0, {'params': [model.alpha_logit], 'lr': args.lr * args.alpha_lr_mult, 'weight_decay': 0.0})
    optimizer = U.make_optimizer(groups, args)
    scheduler = U.make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    meta = dict(model_type='fusion', model_kwargs=kwargs, data=spec, args=vars(args), part='part4',
                params_total=total, params_trainable=trainable)
    run_dir = os.path.join(args.runs_dir, args.run_name)

    epoch_extra = (lambda mdl: {'alpha_rgb': float(mdl.alpha.detach())}) if args.mode == 'attention' else None
    U.fit_classifier(model, loaders, criterion, optimizer, scheduler, args, run_dir, meta, device, epoch_extra)
    if not args.no_test:
        metrics = U.final_test(run_dir, loaders['test'], device, not args.no_amp, args.max_batches)
        if args.mode == 'attention':
            fused, _ = U.load_run(run_dir, device)
            a = float(fused.alpha.detach())
            print(f'Learned alpha (RGB weight): {a:.3f}  -> flow weight {1 - a:.3f}')
        elif 'mean_alpha_rgb' in metrics:
            print(f'Mean per-clip alpha (RGB weight) on test: {metrics["mean_alpha_rgb"]:.3f}')
    if args.export:
        U.export_state_dict(run_dir, args.export)


if __name__ == '__main__':
    main()
