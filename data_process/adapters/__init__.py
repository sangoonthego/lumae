"""Dataset adapters for converting source annotations into canonical format."""

from .base import BaseDatasetAdapter
from .qvhighlights import QVHighlightsAdapter
from .ego4d_nlq import Ego4DNLQAdapter

__all__ = ["BaseDatasetAdapter", "QVHighlightsAdapter", "Ego4DNLQAdapter"]
