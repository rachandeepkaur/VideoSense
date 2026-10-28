"""Model registry: rebuild any VideoSense model from the metadata stored in its checkpoint."""
from models.cnn3d import C3D, R2Plus1DClassifier, R2Plus1DNet, VideoEncoder
from models.fusion import TwoStreamFusion
from models.siamese import VideoSiamese

__all__ = ['C3D', 'R2Plus1DClassifier', 'R2Plus1DNet', 'VideoEncoder', 'TwoStreamFusion',
           'VideoSiamese', 'MODEL_REGISTRY', 'build_model']

MODEL_REGISTRY = {
    'c3d': C3D,
    'r2plus1d': R2Plus1DClassifier,
    'r2plus1d_custom': R2Plus1DNet,
    'siamese': VideoSiamese,
    'fusion': TwoStreamFusion,
}


def build_model(model_type, for_loading=False, **kwargs):
    """for_loading=True skips downloading pretrained weights (the checkpoint overwrites them)."""
    if for_loading and 'pretrained' in kwargs:
        kwargs = dict(kwargs, pretrained=False)
    return MODEL_REGISTRY[model_type](**kwargs)
