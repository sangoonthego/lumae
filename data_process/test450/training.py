"""Train/Validation-only Moment-DETR preflight and nine-run experiment plan."""

import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import time

from data_process.lumae_scale.models import ROOT, atomic_json, digest
from experiments.M1_lumae_ads import cached_runner as runner
from .parent import CANONICAL, COUNTS, M0, SOURCE, save_json
from .release import feature_manifest

SEEDS = (42, 123, 2026)
MODELS = ("moment_detr", "qd_detr", "univtg")
BLOCKED = {
    "qd_detr": {
        "checkpoint": "Drive QR-DETR/videoonly.ckpt (1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL)",
        "blockers": [
            "Stage the candidate bytes, pin the source revision and verify the state dict against QD-DETR.",
            "Verify input projections and stage timeline-aligned SlowFast if required.",
            "Implement and validate a D1500 adapter and a Train-only saliency/loss policy.",
        ],
        "feature_recipe": "Upstream video-only reference: SlowFast 2304 + CLIP 512 + TEF 2; candidate unverified",
    },
    "univtg": {
        "checkpoint": "Drive UniVTG/model_best.ckpt (1bRPJkIvUUyaNOKokCtTM_uRzfTZgZ-hc)",
        "blockers": [
            "Stage and hash the candidate and opt.json, then pin and strict-load the corresponding UniVTG code.",
            "Verify the candidate feature streams and stage aligned SlowFast if required.",
            "Validate the D1500 text layout, seconds conversion and Train-only weak saliency/loss policy.",
        ],
        "feature_recipe": "QVHighlights reference: SlowFast 2304 + CLIP 512 + TEF 2; candidate unverified",
    },
}


def recipe(parent, seed):
    if seed not in SEEDS:
        raise ValueError("Seed is outside the registered D1500 matrix")
    return runner.read(runner.CONFIG) | {
        "dataset": "lumae_ads_d1500", "canonical_sha256": CANONICAL,
        "split_sha256": {s: parent.hashes[f"{s}.jsonl"] for s in ("train", "val")},
        "counts": {s: COUNTS[s] for s in ("train", "val")},
        "seed": seed, "deterministic": True, "dataloader_workers": 0,
    }


def matrix(parent, work):
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            entry = {
                "model": model, "seed": seed, "selection_split": "val",
                "planned_checkpoint": str(work / "runs" / model / f"seed_{seed}" / "best.ckpt"),
                "execution": "NOT_RUN",
            }
            if model == "moment_detr":
                entry |= {
                    "checkpoint": str(SOURCE / "assets/M0_pretrained.ckpt"), "checkpoint_sha256": M0,
                    "training_config": recipe(parent, seed),
                    "feature_recipe": "Frozen CLIP ViT-B/32 token states and 2-second RGB clips; TEF added on load",
                    "preflight": "NOT_RUN",
                }
                result = work / "preflight" / f"moment_detr_seed_{seed}.json"
                if result.is_file():
                    entry["preflight"] = runner.read(result)["status"]
            else:
                entry |= BLOCKED[model] | {"preflight": "BLOCKED", "training_config": None}
            checkpoint = Path(entry["planned_checkpoint"])
            if checkpoint.is_file() and checkpoint.with_name("training_history.json").is_file():
                entry["execution"] = "CHECKPOINT_AND_HISTORY_PRESENT"
            rows.append(entry)
    return {"split_seed": parent.seed, "training_seeds": SEEDS, "runs": rows}


def prepare(parent, work, seed):
    features = parent.features(("train", "val"))
    root = Path(os.path.commonpath([f.path.parent for group in features.values() for f in group.values()]))
    for group in features.values():
        for feature in group.values():
            feature.array()
    manifest = feature_manifest(features, lambda f: f.path.relative_to(root).as_posix())
    index = work / "training/train_val_features.json"
    save_json(index, manifest)
    cfg = recipe(parent, seed)
    datasets, identity, exceptions = runner.preflight(
        parent.frozen / "train.jsonl", parent.frozen / "val.jsonl", index, root,
        SOURCE / "upstream", SOURCE / "assets/M0_pretrained.ckpt", cfg,
    )
    for field, kind in (("vid", "video"), ("qid", "text")):
        expected = {str(row[field]) for dataset in datasets.values() for row in dataset.data}
        if set(manifest[kind]) != expected:
            raise ValueError("Training index must cover Train/Val exactly")
    return datasets, identity, exceptions


def forward_check(datasets, identity):
    import torch
    from moment_detr.start_end_dataset import prepare_batch_inputs, start_end_collate

    model, criterion, opt = runner.initialize_training_model(
        SOURCE / "upstream", SOURCE / "assets/M0_pretrained.ckpt", identity["recipe"]["seed"], "cpu",
    )
    model.eval()
    with torch.inference_mode():
        meta, batch = start_end_collate([datasets["train"][0], datasets["val"][0]])
        inputs, targets = prepare_batch_inputs(batch, torch.device("cpu"))
        output = model(**inputs)
        losses = criterion(output, targets)
    if not all(torch.isfinite(torch.as_tensor(value)).all().item()
               for value in (output["pred_spans"], *losses.values())):
        raise ValueError("Nonfinite model output or loss in preflight")
    return {
        "model_tensor_count": len(model.state_dict()), "batch_qids": [row["qid"] for row in meta],
        "video_input_dim": opt.v_feat_dim, "text_input_dim": opt.t_feat_dim,
        "output_shape": list(output["pred_spans"].shape), "loss_keys": sorted(losses),
    }


def preflight(parent, work, seed, execute=False, resume=False):
    import torch

    if execute:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch runtime")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    datasets, identity, exceptions = prepare(parent, work, seed)
    smoke = forward_check(datasets, identity)
    summary = {
        "status": "PASS", "strict_load": True, "cpu_forward_and_loss": smoke,
        "training_config": identity["recipe"], "runtime": identity["runtime"],
        "feature_manifest_sha256": identity["feature_manifest_sha256"], "dtype_exceptions": exceptions,
        "runner_sha256": digest(Path(runner.__file__)), "m0_sha256": M0,
        "upstream_evaluator_sha256": identity["evaluator_sha256"],
        "repo_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "cuda_available": torch.cuda.is_available(),
    }
    if execute:
        output = work / "runs/moment_detr" / f"seed_{seed}"
        identity["execution_device"] = torch.cuda.get_device_name(0)
        identity["training_cli_sha256"] = digest(Path(__file__))
        from data_process.lumae_scale.acquisition import run_lock

        with run_lock(work / "runs/moment_detr" / f"seed_{seed}.lock"):
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            history = runner.train(datasets, identity, SOURCE / "upstream",
                                   SOURCE / "assets/M0_pretrained.ckpt", output, resume)
            atomic_json(output / "completion.json", {
                "status": "COMPLETE", "seed": seed, "started_at_utc": started_at,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_elapsed_seconds": time.perf_counter() - started,
                "training_epoch_seconds": sum(row["elapsed_seconds"] for row in history),
                "best_epoch": next(row["epoch"] for row in reversed(history) if row["is_best"]),
                "best_checkpoint_sha256": digest(output / "best.ckpt"),
                "device": identity["execution_device"], "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                "selection": "val R1@0.5, then val mIoU",
            })
        return summary | {"execution": "COMPLETE", "output": str(output)}
    atomic_json(work / "preflight" / f"moment_detr_seed_{seed}.json", summary)
    return summary
