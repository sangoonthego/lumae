"""Machine-readable representation and plan for future Colab GPU jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class GpuJobType(str, Enum):
    """Supported Colab GPU job types."""
    VIDEO_FEATURE_EXTRACTION = "VIDEO_FEATURE_EXTRACTION"
    TEXT_FEATURE_EXTRACTION = "TEXT_FEATURE_EXTRACTION"
    TRAIN_STAGE_A = "TRAIN_STAGE_A"
    EVAL_STAGE_A_EGO4D = "EVAL_STAGE_A_EGO4D"
    EVAL_STAGE_A_QVHIGHLIGHTS = "EVAL_STAGE_A_QVHIGHLIGHTS"


class GpuJobStatus(str, Enum):
    """Job lifecycle statuses."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class GpuJob:
    """Individual GPU execution unit to be run in Colab."""
    job_id: str
    job_type: GpuJobType | str
    inputs: list[str]
    outputs: list[str]
    dependencies: list[str] = field(default_factory=list)
    status: GpuJobStatus | str = GpuJobStatus.PENDING
    config_reference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        j_type = self.job_type.value if isinstance(self.job_type, GpuJobType) else str(self.job_type)
        status_val = self.status.value if isinstance(self.status, GpuJobStatus) else str(self.status)
        return {
            "job_id": self.job_id,
            "job_type": j_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "status": status_val,
            "config_reference": self.config_reference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GpuJob:
        return cls(
            job_id=data["job_id"],
            job_type=GpuJobType(data["job_type"]) if data["job_type"] in GpuJobType._value2member_map_ else data["job_type"],
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            dependencies=list(data.get("dependencies", [])),
            status=GpuJobStatus(data.get("status", GpuJobStatus.PENDING.value)),
            config_reference=data.get("config_reference", {}),
        )


@dataclass
class GpuJobManifest:
    """DAG of planned Colab GPU jobs."""
    version: str = "1.0"
    target_hardware: str = "Google Colab T4"
    jobs: list[GpuJob] = field(default_factory=list)

    def add_job(self, job: GpuJob) -> None:
        self.jobs.append(job)

    def validate_dependencies(self) -> tuple[bool, list[str]]:
        """Validate that all referenced dependencies exist and form an acyclic graph."""
        errors: list[str] = []
        known_ids = {j.job_id for j in self.jobs}

        for j in self.jobs:
            for dep in j.dependencies:
                if dep not in known_ids:
                    errors.append(f"Job '{j.job_id}' references unknown dependency '{dep}'")

        # Cycle detection via topological sort
        in_degree: dict[str, int] = {j.job_id: 0 for j in self.jobs}
        adj: dict[str, list[str]] = {j.job_id: [] for j in self.jobs}
        for j in self.jobs:
            for dep in j.dependencies:
                if dep in known_ids:
                    adj[dep].append(j.job_id)
                    in_degree[j.job_id] += 1

        queue = [jid for jid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited < len(self.jobs):
            errors.append("Cyclic dependency detected among GPU jobs")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target_hardware": self.target_hardware,
            "total_jobs": len(self.jobs),
            "jobs": [j.to_dict() for j in self.jobs],
        }

    def save(self, path: Path | str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> GpuJobManifest:
        dest = Path(path)
        data = json.loads(dest.read_text(encoding="utf-8"))
        jobs = [GpuJob.from_dict(j) for j in data.get("jobs", [])]
        return cls(
            version=data.get("version", "1.0"),
            target_hardware=data.get("target_hardware", "Google Colab T4"),
            jobs=jobs,
        )
