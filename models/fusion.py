"""Two-stream RGB + optical-flow models (Part 4).

One class covers the whole ablation and all fusion strategies, so that every variant shares the
same encoders, heads, data and training code and only the fusion differs:

mode='rgb'        : RGB stream only                                    (ablation)
mode='flow'       : flow stream only                                   (ablation)
mode='concat'     : late fusion, concatenate embeddings -> classifier  (assignment default)
mode='avg'        : late fusion, average the two streams' logits
mode='attention'  : learned scalar alpha: alpha * z_rgb + (1 - alpha) * z_flow   (Exercise 23)
mode='gated'      : alpha computed per clip from both embeddings (content-adaptive attention)
mode='cross_attn' : flow tokens attend to RGB tokens and vice versa, then concat
"""
import torch
import torch.nn as nn

from models.cnn3d import VideoEncoder

MODES = ('rgb', 'flow', 'concat', 'avg', 'attention', 'gated', 'cross_attn')


class TwoStreamFusion(nn.Module):
    """Input: rgb (B,3,T,H,W), flow (B,2,T,H,W). Output: logits (B, num_classes)."""

    def __init__(self, num_classes=101, mode='concat', backbone='c3d', embed_dim=512, dropout=0.5,
                 pretrained=False, batch_norm=False, attn_heads=4):
        super().__init__()
        assert mode in MODES, f'mode must be one of {MODES}'
        self.mode = mode
        self.use_rgb, self.use_flow = mode != 'flow', mode != 'rgb'
        if self.use_rgb:
            self.rgb_encoder = VideoEncoder(backbone, 3, pretrained, batch_norm)
            self.rgb_head = nn.Sequential(nn.Linear(self.rgb_encoder.out_dim, embed_dim), nn.ReLU())
        if self.use_flow:
            # Kinetics-pretrained flow stream = RGB weights adapted to 2 channels (cross-modality init)
            self.flow_encoder = VideoEncoder(backbone, 2, pretrained, batch_norm)
            self.flow_head = nn.Sequential(nn.Linear(self.flow_encoder.out_dim, embed_dim), nn.ReLU())

        if mode in ('rgb', 'flow'):
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        elif mode == 'concat':
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * embed_dim, num_classes))
        elif mode == 'avg':
            self.rgb_cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
            self.flow_cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        elif mode in ('attention', 'gated'):
            # LayerNorm fixes each embedding's scale. Without it the network could make alpha
            # meaningless by shrinking or inflating one embedding instead of changing alpha.
            self.norm_rgb, self.norm_flow = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
            if mode == 'attention':
                self.alpha_logit = nn.Parameter(torch.zeros(1))      # sigmoid(0) = 0.5 at start
            else:
                self.gate = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim // 4), nn.ReLU(),
                                          nn.Linear(embed_dim // 4, 1))
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)                   # alpha = 0.5 at start
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        elif mode == 'cross_attn':
            c = self.rgb_encoder.map_channels
            self.rgb_tok, self.flow_tok = nn.Linear(c, embed_dim), nn.Linear(c, embed_dim)
            self.flow_to_rgb = nn.MultiheadAttention(embed_dim, attn_heads, dropout=0.1, batch_first=True)
            self.rgb_to_flow = nn.MultiheadAttention(embed_dim, attn_heads, dropout=0.1, batch_first=True)
            self.ln_rgb, self.ln_flow = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * embed_dim, num_classes))
        self.last_alpha = None

    # ---- pieces
    @property
    def alpha(self):
        """Learned weight of the RGB stream ('attention' mode)."""
        return torch.sigmoid(self.alpha_logit) if self.mode == 'attention' else None

    def get_stream_embeddings(self, rgb, flow):
        z_rgb = self.rgb_head(self.rgb_encoder(rgb)) if self.use_rgb else None
        z_flow = self.flow_head(self.flow_encoder(flow)) if self.use_flow else None
        return z_rgb, z_flow

    def _cross_attention(self, rgb, flow):
        def tok(fmap, proj):
            return proj(fmap.flatten(2).transpose(1, 2))                  # (B, t*h*w, embed)
        r = tok(self.rgb_encoder.feature_map(rgb), self.rgb_tok)
        f = tok(self.flow_encoder.feature_map(flow), self.flow_tok)
        r2 = self.ln_rgb(r + self.flow_to_rgb(r, f, f, need_weights=False)[0])   # RGB queries flow
        f2 = self.ln_flow(f + self.rgb_to_flow(f, r, r, need_weights=False)[0])  # flow queries RGB
        return torch.cat([r2.mean(1), f2.mean(1)], dim=1)

    def get_embedding(self, rgb, flow):
        """Fused representation fed to the final classifier (used for t-SNE)."""
        if self.mode == 'cross_attn':
            return self._cross_attention(rgb, flow)
        z_rgb, z_flow = self.get_stream_embeddings(rgb, flow)
        if self.mode == 'rgb':
            return z_rgb
        if self.mode == 'flow':
            return z_flow
        if self.mode in ('concat', 'avg'):
            return torch.cat([z_rgb, z_flow], dim=1)
        z_rgb, z_flow = self.norm_rgb(z_rgb), self.norm_flow(z_flow)
        if self.mode == 'attention':
            a = self.alpha.expand(z_rgb.size(0), 1)
        else:
            a = torch.sigmoid(self.gate(torch.cat([z_rgb, z_flow], dim=1)))
        self.last_alpha = a.detach().squeeze(1).float()
        return a * z_rgb + (1 - a) * z_flow

    def forward(self, rgb, flow):
        if self.mode == 'avg':
            z_rgb, z_flow = self.get_stream_embeddings(rgb, flow)
            return 0.5 * (self.rgb_cls(z_rgb) + self.flow_cls(z_flow))
        return self.classifier(self.get_embedding(rgb, flow))

    @property
    def cam_layer(self):
        """Grad-CAM target: last conv of the RGB stream (or the flow stream in flow-only mode)."""
        return (self.rgb_encoder if self.use_rgb else self.flow_encoder).cam_layer
