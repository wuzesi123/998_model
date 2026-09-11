from .schema import AdapterCapabilities, CanonicalExperiment, union_feature_space
from .base import BaseReactionAdapter
from .kineticolor import KineticolorAdapter
from .crystalcv import CrystalCVAdapter
from .heinsight4 import HeinSight4Adapter

ADAPTERS = {
    "kineticolor": KineticolorAdapter,
    "crystalcv": CrystalCVAdapter,
    "heinsight4": HeinSight4Adapter,
}


def make_adapter(name, root):
    key = str(name).lower().strip()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown adapter {name!r}; choose from {sorted(ADAPTERS)}")
    return ADAPTERS[key](root)
