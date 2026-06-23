from .baselines import build_baseline
from .catnet import CATNet, build_catnet

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "catnet":
        return build_catnet(num_classes)
    return build_baseline(name, num_classes)
