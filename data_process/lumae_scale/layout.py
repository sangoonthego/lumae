"""Explicit output layout for incremental dataset builds; D200 defaults remain."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .models import ROOT, SEED


@dataclass(frozen=True)
class BuildLayout:
    root: Path = ROOT
    version: str = "lumae_ads_d500"
    base_version: str = "lumae_ads_d200"
    base_count: int = 200
    new_count: int = 300
    seed: int = SEED

    def __post_init__(self):
        if (self.version, self.base_version, self.base_count, self.new_count, self.seed) != (
                "lumae_ads_d500", "lumae_ads_d200", 200, 300, 20260930):
            raise ValueError("Main scaling experiment requires D200 + 300 with fixed benchmark and seed")

    @property
    def work(self): return self.root / "local_data/intermediate" / self.version
    @property
    def dataset(self): return self.root / "local_data/datasets" / self.version
    @property
    def base(self): return self.root / "local_data/datasets" / self.base_version
    @property
    def accepted(self): return self.work / "accepted"
    @property
    def reports(self): return self.root / "local_data/reports" / self.version
    @property
    def selection(self): return self.work / "source_selection.json"
    @property
    def ledger(self): return self.work / "build_ledger.jsonl"
    @property
    def visual_cache(self): return self.root / "local_data/cache/lumae_visual_index"
    @property
    def videos(self): return self.root / "local_data/raw/adsqa/videos"
    @property
    def handoff(self): return self.root / "local_data/manifests" / f"{self.version}_training_handoff.json"
