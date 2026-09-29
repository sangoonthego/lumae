"""Abstract base adapter for dataset ingestion and canonicalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..io.jsonl import read_jsonl
from ..schemas.temporal_sample import (
    CanonicalTemporalSample,
    ReasonCode,
    RejectedSampleRecord,
)


class BaseDatasetAdapter(ABC):
    """Base class for source dataset adapters."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def convert_record(
        self,
        raw_record: dict[str, Any],
        index: int,
    ) -> tuple[CanonicalTemporalSample | None, RejectedSampleRecord | None]:
        """Convert a single raw annotation record into a CanonicalTemporalSample.

        Returns:
            (sample, None) on success.
            (None, rejected_record) on failure/invalid data.
        """
        pass

    def process_file(
        self,
        input_path: Path | str,
    ) -> tuple[list[CanonicalTemporalSample], list[RejectedSampleRecord]]:
        """Process an entire source annotation file.

        Returns:
            (valid_samples, rejected_records)
        """
        valid_samples: list[CanonicalTemporalSample] = []
        rejected_records: list[RejectedSampleRecord] = []

        for idx, record in enumerate(read_jsonl(input_path)):
            sample, rejected = self.convert_record(record, idx)
            if sample is not None:
                valid_samples.append(sample)
            elif rejected is not None:
                rejected_records.append(rejected)
            else:
                # Should not happen if convert_record is implemented properly
                rejected_records.append(
                    RejectedSampleRecord(
                        source=self.name,
                        original_id=record.get("qid") or record.get("id") or str(idx),
                        reason_code=ReasonCode.UNSUPPORTED_SOURCE_RECORD,
                        reason="Adapter returned neither sample nor rejection record",
                        raw_reference=str(record)[:200],
                    )
                )

        return valid_samples, rejected_records
