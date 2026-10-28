"""3D CNN architectures for VideoSense (Part 2).

Contents
--------
C3D                 : the assignment's C3D (Tran et al., 2015), optional BatchNorm, any input channels.
R2Plus1DBlock       : the assignment's (2+1)D residual block (spatial 1x3x3 conv, then temporal 3x1x1 conv).
R2Plus1DNet         : a small R(2+1)D built from R2Plus1DBlock, trained from scratch (optional comparison).
VideoEncoder        : shared spatiotemporal feature extractor ('c3d' or torchvision 'r2plus1d'),
                      reused by the Siamese network (Part 3) and the two-stream model (Part 4).
R2Plus1DClassifier  : torchvision R(2+1)D-18 pretrained on Kinetics-400 with a new classifier head.
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- C3D
class C3D(nn.Module):
    """C3D for video classification.

    Input : (B, in_channels, T=16, H=112, W=112)
    Output: (B, num_classes)

    Tensor shapes after each pooling layer (T=16, 112x112 input):
        pool1 (1,2,2) -> (B,  64, 16, 56, 56)   temporal size kept: early layers see fine motion
        pool2 (2,2,2) -> (B, 128,  8, 28, 28)
        pool3 (2,2,2) -> (B, 256,  4, 14, 14)
        pool4 (2,2,2) -> (B, 512,  2,  7,  7)
        pool5 (2,2,2) -> (B, 512,  1,  3,  3)   all 16 input frames are summarized into 1 time step
    """

    def __init__(self, num_classes=101, dropout=0.5, in_channels=3, batch_norm=False):
        super().__init__()
        cfg = [(in_channels, 64, 1), ('P', (1, 2, 2)),
               (64, 128, 1), ('P', (2, 2, 2)),
               (128, 256, 2), ('P', (2, 2, 2)),
               (256, 512, 2), ('P', (2, 2, 2)),
               (512, 512, 2), ('P', (2, 2, 2))]
        layers = []
        for item in cfg:
            if item[0] == 'P':
                layers.append(nn.MaxPool3d(kernel_size=item[1], stride=item[1]))
                continue
            c_in, c_out, n_conv = item
            for i in range(n_conv):
                layers.append(nn.Conv3d(c_in if i == 0 else c_out, c_out, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm3d(c_out))
                layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)
        self.feat_dim = 512 * 1 * 3 * 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_embedding(self, x):
        """4096-d embedding after fc1 + ReLU (used for t-SNE)."""
        return self.classifier[:3](self.features(x))

    @property
    def cam_layer(self):
        """Last Conv3d: output (B, 512, 2, 7, 7) for 16x112x112 input. Used by Grad-CAM."""
        return [m for m in self.features if isinstance(m, nn.Conv3d)][-1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- R(2+1)D
class R2Plus1DBlock(nn.Module):
    """One R(2+1)D residual block (from the assignment): 1x3x3 spatial conv, then 3x1x1 temporal conv.

    Factorizing a 3x3x3 conv this way adds an extra non-linearity between space and time and
    makes optimization easier (Tran et al., 2018). Note: the paper also sizes the intermediate
    channels so that the parameter count matches a full 3D conv; this simplified block uses out_ch.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
        self.temporal = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                      padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch))

    def forward(self, x):
        return torch.relu(self.temporal(self.spatial(x)) + self.shortcut(x))


class R2Plus1DNet(nn.Module):
    """Small R(2+1)D built from R2Plus1DBlock, for a from-scratch comparison with C3D."""

    def __init__(self, num_classes=101, in_channels=3, dropout=0.5, blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        stages, c_in = [], 64
        for c_out, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            for b in range(blocks_per_stage):
                stages.append(R2Plus1DBlock(c_in, c_out, stride if b == 0 else 1))
                c_in = c_out
        self.layers = nn.Sequential(*stages)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.fc(self.get_embedding(x))

    def get_embedding(self, x):
        return self.pool(self.layers(self.stem(x)))

    @property
    def cam_layer(self):
        return self.layers[-1]


# ----------------------------------------------------------------------------- shared encoder
def adapt_first_conv_weight(weight, in_channels):
    """Cross-modality initialization (Wang et al., TSN 2016): average the RGB filters over the input
    channels and replicate the average for `in_channels` (e.g. 2 flow channels), divided by
    in_channels, exactly as in the assignment's TwoStreamFusion template."""
    w = weight.mean(dim=1, keepdim=True)
    return w.repeat(1, in_channels, *([1] * (weight.dim() - 2))) / in_channels


class VideoEncoder(nn.Module):
    """Spatiotemporal feature extractor shared by the Siamese and two-stream models.

    backbone='c3d'      : C3D conv layers, output flattened pool5 (4608-d), feature map (512,1,3,3)
    backbone='r2plus1d' : torchvision R(2+1)D-18 (optionally Kinetics-400 pretrained), output
                          global-average-pooled 512-d, feature map (512, T/8, 7, 7)

    feature_map(x) -> conv features (B, C, t, h, w)   [cross-attention, Grad-CAM]
    forward(x)     -> pooled vector (B, out_dim)
    """

    def __init__(self, backbone='c3d', in_channels=3, pretrained=False, batch_norm=False):
        super().__init__()
        self.backbone = backbone
        if backbone == 'c3d':
            if pretrained:
                raise ValueError('No public pretrained C3D in torchvision: use backbone="r2plus1d" '
                                 'for pretraining, or initialize from your own C3D checkpoint.')
            self.features = C3D(num_classes=1, in_channels=in_channels, batch_norm=batch_norm).features
            self.pool = nn.Flatten()
            self.out_dim, self.map_channels = 512 * 1 * 3 * 3, 512
        elif backbone == 'r2plus1d':
            from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
            net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
            if in_channels != 3:
                old = net.stem[0]
                new = nn.Conv3d(in_channels, old.out_channels, old.kernel_size, old.stride,
                                old.padding, bias=False)
                with torch.no_grad():
                    new.weight.copy_(adapt_first_conv_weight(old.weight, in_channels))
                net.stem[0] = new
            self.features = nn.Sequential(net.stem, net.layer1, net.layer2, net.layer3, net.layer4)
            self.pool = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
            self.out_dim, self.map_channels = 512, 512
        else:
            raise ValueError(f'unknown backbone {backbone}')

    def feature_map(self, x):
        return self.features(x)

    def forward(self, x):
        return self.pool(self.features(x))

    @property
    def cam_layer(self):
        if self.backbone == 'c3d':
            return [m for m in self.features if isinstance(m, nn.Conv3d)][-1]
        return self.features[-1]                                    # layer4


def extract_encoder_state(state_dict, prefix=None):
    """Pull the 'features.*' weights out of any VideoSense checkpoint.

    Works for C3D ('features.*'), R2Plus1DClassifier ('encoder.features.*'), VideoSiamese
    ('encoder.features.*') and TwoStreamFusion ('rgb_encoder.features.*' / 'flow_encoder.features.*').
    """
    candidates = [prefix] if prefix is not None else ['', 'encoder.', 'rgb_encoder.', 'flow_encoder.']
    for p in candidates:
        sub = {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p + 'features.')}
        if sub:
            return sub
    raise KeyError('no encoder weights found in checkpoint')


def load_encoder_weights(encoder, state_dict, prefix=None):
    """Load encoder weights; if the checkpoint has a different number of input channels
    (e.g. RGB weights into a flow encoder), the first conv is cross-modality adapted."""
    sub = extract_encoder_state(state_dict, prefix)
    own = encoder.state_dict()
    for k, v in sub.items():
        if k in own and own[k].shape != v.shape and v.dim() == 5 and own[k].shape[1] != v.shape[1]:
            sub[k] = adapt_first_conv_weight(v, own[k].shape[1])
    missing, unexpected = encoder.load_state_dict(sub, strict=False)
    missing = [m for m in missing if not m.endswith('num_batches_tracked')]
    if missing or unexpected:
        raise RuntimeError(f'encoder weight mismatch. missing={missing[:5]} unexpected={unexpected[:5]}')


class R2Plus1DClassifier(nn.Module):
    """torchvision R(2+1)D-18 + new classifier head (Part 2, Exercise 9).

    freeze_backbone=True trains only the head ("linear probe"). The backbone is then also kept in
    eval mode, so its BatchNorm layers keep the Kinetics running statistics instead of being
    overwritten by statistics from small UCF101 batches.
    train_layer4=True additionally fine-tunes the last stage (partial fine-tuning).
    """

    def __init__(self, num_classes=101, pretrained=True, freeze_backbone=True, train_layer4=False,
                 dropout=0.0):
        super().__init__()
        self.encoder = VideoEncoder('r2plus1d', 3, pretrained=pretrained)
        self.freeze_backbone, self.train_layer4 = freeze_backbone, train_layer4
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if train_layer4:
                for p in self.encoder.features[-1].parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone and mode:
            self.encoder.eval()
            if self.train_layer4:
                self.encoder.features[-1].train()
        return self

    def forward(self, x):
        return self.fc(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)

    @property
    def cam_layer(self):
        return self.encoder.cam_layer
