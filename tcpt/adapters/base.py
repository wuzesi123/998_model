from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schema import AdapterCapabilities, CanonicalExperiment


class BaseReactionAdapter(ABC):
    dataset_name: str = "unknown"
    capabilities = AdapterCapabilities(temporal=False)

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def discover(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def convert(self) -> list[CanonicalExperiment]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "root": str(self.root),
            "capabilities": self.capabilities.__dict__,
            "n_discovered": len(self.discover()) if self.root.exists() else 0,
        }
