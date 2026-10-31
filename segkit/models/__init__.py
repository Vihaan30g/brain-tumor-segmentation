from segkit.registry import Registry
from segkit.models.transunet import TransUNet

MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register("transunet", TransUNet)


def build_model(cfg, **overrides):
    """Build the model from cfg.model. `overrides` replace config values
    (e.g. build_model(cfg, pretrained=False) to avoid downloading weights for evaluation)."""
    model_name = cfg.model.name
    model_cls = MODEL_REGISTRY.get(model_name)
    kwargs = {k: v for k, v in dict(cfg.model).items() if k != "name"}
    kwargs.update(overrides)
    return model_cls(**kwargs)
