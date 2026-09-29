"""Tests for FeatureManifest and GpuJobManifest."""

from pathlib import Path
import tempfile

from data_process.manifests.features import generate_feature_manifest, FeatureManifest
from data_process.manifests.gpu_jobs import (
    GpuJob,
    GpuJobManifest,
    GpuJobStatus,
    GpuJobType,
)
from data_process.schemas.temporal_sample import CanonicalTemporalSample


def test_generate_feature_manifest():
    samples = [
        CanonicalTemporalSample(
            qid=1, vid="v1", query="Query 1", duration=10.0, relevant_windows=[[1.0, 3.0]]
        ),
        CanonicalTemporalSample(
            qid=2, vid="v1", query="Query 2", duration=10.0, relevant_windows=[[4.0, 6.0]]
        ),
        CanonicalTemporalSample(
            qid=3, vid="v2", query="Query 3", duration=20.0, relevant_windows=[[2.0, 8.0]]
        ),
    ]

    manifest = generate_feature_manifest(
        samples,
        dataset_name="test_dataset",
        video_feature_dir="Features/test/video",
        text_feature_dir="Features/test/text",
    )

    assert manifest.dataset_name == "test_dataset"
    assert len(manifest.items) == 2  # 2 unique vids (v1, v2)
    v1_item = next(item for item in manifest.items if item.vid == "v1")
    assert len(v1_item.queries) == 2  # 2 queries for v1
    assert v1_item.video_feature.path == "Features/test/video/v1.npz"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "manifest.json"
        manifest.save(out_file)
        loaded = FeatureManifest.load(out_file)
        assert loaded.dataset_name == "test_dataset"
        assert len(loaded.items) == 2


def test_gpu_job_manifest_dag_validation():
    manifest = GpuJobManifest()

    job1 = GpuJob(
        job_id="extract_features",
        job_type=GpuJobType.VIDEO_FEATURE_EXTRACTION,
        inputs=["data/videos.json"],
        outputs=["features/video/"],
    )
    job2 = GpuJob(
        job_id="train_stage_a",
        job_type=GpuJobType.TRAIN_STAGE_A,
        inputs=["features/video/"],
        outputs=["checkpoints/m1/model_best.ckpt"],
        dependencies=["extract_features"],
    )
    manifest.add_job(job1)
    manifest.add_job(job2)

    ok, errors = manifest.validate_dependencies()
    assert ok is True
    assert len(errors) == 0


def test_gpu_job_manifest_unknown_dependency():
    manifest = GpuJobManifest()
    job = GpuJob(
        job_id="eval_stage_a",
        job_type=GpuJobType.EVAL_STAGE_A_EGO4D,
        inputs=[],
        outputs=[],
        dependencies=["non_existent_job"],
    )
    manifest.add_job(job)
    ok, errors = manifest.validate_dependencies()
    assert ok is False
    assert any("unknown dependency" in e for e in errors)


def test_gpu_job_manifest_cyclic_dependency():
    manifest = GpuJobManifest()
    job_a = GpuJob(job_id="job_a", job_type="custom", inputs=[], outputs=[], dependencies=["job_b"])
    job_b = GpuJob(job_id="job_b", job_type="custom", inputs=[], outputs=[], dependencies=["job_a"])
    manifest.add_job(job_a)
    manifest.add_job(job_b)

    ok, errors = manifest.validate_dependencies()
    assert ok is False
    assert any("cyclic" in e.lower() for e in errors)
