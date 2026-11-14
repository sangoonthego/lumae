"""Freeze and package the LUMAE D1500 Test-300 interim benchmark for Drive and Colab."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import re
import sys
import zipfile

from experiments.M1_lumae_ads import cached_runner as runner

from .annotation import AnnotationStore, progress as compute_progress
from .evaluation import load_predictions, metrics as compute_metrics
from .parent import (
    CANONICAL,
    COUNTS,
    INVENTORY,
    M0,
    ROOT,
    SEAL,
    SOURCE,
    Parent,
    TrainingParent,
    duration_distribution,
    jsonl_bytes,
    read_json,
    require_hash,
    sha_bytes,
    valid_window,
)
from .release import EXTRACTOR_HASHES, WEIGHTS_HASH, quality as release_quality, text_path
from .training import BLOCKED, MODELS, SEEDS

VERSION = "LUMAE_D1500_TEST300_AI_V1"
FIXED_ZIP_DT = (2026, 11, 14, 0, 0, 0)
DRIVE_ARTIFACT_ROOT = "CV_Lumae/artifacts/d1500_test300_ai_v1"
DRIVE_LINKS = {
    "d500_historical_features_folder": "https://drive.google.com/drive/folders/1cTUQP7DVdexC89FJWeWUNoncbBRVhrQo",
    "m0_pretrained_checkpoint_file": "https://drive.google.com/file/d/1tY354GpxxplRVXQ0yS0Laaus0Am6K-8i/view?usp=drivesdk",
}
FORBIDDEN_LOCAL_PATH_RE = re.compile(r"(?:(?<![A-Za-z])[A-Za-z]:(?:\\|/(?!/))|/(?:Users|home)/)")


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path: pathlib.Path, value: object) -> None:
    _write_bytes(path, _json_bytes(value))


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_base_feature_index(parent: Parent) -> dict:
    """Build portable relative feature mapping for all 3,000 frozen D1500 base NPZ tensors."""
    require_hash(parent.feature_inventory, INVENTORY)
    manifest = read_json(parent.feature_inventory)
    all_rows = [row for split in ("train", "val", "test") for row in parent.rows[split]]
    expected_vids = {r["vid"] for r in all_rows}
    expected_qids = {str(r["qid"]) for r in all_rows}
    if len(expected_vids) != 1500 or len(expected_qids) != 1500:
        raise ValueError("Parent split does not contain exactly 1500 unique videos and queries")

    items = {"video": {}, "text": {}}
    by_source = Counter()
    seen_paths = set()
    for entry in manifest["entries"]:
        if entry.get("category") != "shared_cached_features":
            continue
        ident = entry["logical_artifact_name"]
        archive = entry.get("archive")
        member = entry.get("archive_member")
        if member:
            if not member.startswith("features/"):
                raise ValueError(f"Unexpected archive_member prefix: {member}")
            rel = member[len("features/") :]
            source_bundle = archive
        else:
            dest = entry.get("intended_drive_destination") or ""
            if not dest.startswith("existing Drive "):
                raise ValueError(f"Unexpected D500 destination: {dest}")
            rel = dest[len("existing Drive ") :]
            source_bundle = "DRIVE_D500"
        if rel in seen_paths or "\\" in rel or ":" in rel or rel.startswith("/"):
            raise ValueError(f"Non-portable or duplicate relative feature path: {rel}")
        seen_paths.add(rel)

        kind = "text" if rel.split("/")[-2] == "clip_text_features" else "video"
        if ident in items[kind]:
            raise ValueError(f"Duplicate base feature entry for {kind} {ident}")
        record = {
            "relative_path": f"features/{rel}",
            "manifest_path": rel,
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
            "dataset_version": entry["dataset_version"],
            "source_bundle": source_bundle,
            "archive_member": member,
        }
        if entry.get("drive_match"):
            record["drive_id"] = entry["drive_match"]["id"]
            record["drive_folder_id"] = entry["drive_match"]["folder_id"]
        items[kind][ident] = record
        by_source[(entry["dataset_version"], source_bundle, kind)] += 1

    if set(items["video"]) != expected_vids or set(items["text"]) != expected_qids:
        raise ValueError("Base feature index does not exactly cover all 1500 videos and 1500 queries")

    return {
        "schema_version": "1.0",
        "parent_canonical_sha256": CANONICAL,
        "feature_inventory_sha256": INVENTORY,
        "counts": {
            "video": len(items["video"]),
            "text": len(items["text"]),
            "total": len(items["video"]) + len(items["text"]),
        },
        "source_breakdown": {
            f"{ver}/{bundle}/{kind}": count for (ver, bundle, kind), count in sorted(by_source.items())
        },
        "package_c_required": False,
        "package_c_rationale": (
            "All 3,000 base cached CLIP NPZs are already verified present across Drive D500 "
            "(1,000), lumae_d1000_colab.zip (1,000), and lumae_d1500_incremental_colab.zip (1,000)."
        ),
        "video": dict(sorted(items["video"].items())),
        "text": dict(sorted(items["text"].items())),
    }


def collect_test300_state(parent: Parent, work_dir: pathlib.Path) -> dict:
    """Validate and assemble all accepted Test-300 rows, provenance, and CLIP text features."""
    audit_report = parent.audit()
    store = AnnotationStore(parent, work_dir)
    new_rows, sidecar, states = store.accepted(0.70)
    prog = compute_progress(new_rows, sidecar, states)

    if len(new_rows) != 150 or len(sidecar) != 150:
        raise ValueError(f"Expected 150 new accepted queries, got rows={len(new_rows)}, sidecar={len(sidecar)}")

    new_vids = {r["vid"] for r in new_rows}
    orig_test_rows = parent.rows["test"]
    orig_test_vids = {r["vid"] for r in orig_test_rows}
    if len(new_vids) != 75 or not new_vids.issubset(orig_test_vids):
        raise ValueError("Accepted new queries must cover exactly 75 videos belonging to frozen Test-150")

    per_vid_new = Counter(r["vid"] for r in new_rows)
    if set(per_vid_new.values()) != {2}:
        raise ValueError("Each completed Test-300 video must have exactly 2 accepted additional queries")

    existing_qids = {
        str(r["qid"])
        for split in ("train", "val", "test")
        for r in parent.rows[split]
    }
    new_qids = [str(r["qid"]) for r in new_rows]
    if len(set(new_qids)) != 150 or (set(new_qids) & existing_qids):
        raise ValueError("New query IDs must be unique and disjoint from parent Train/Val/Test")

    for row, record in zip(new_rows, sidecar):
        if len(row["relevant_windows"]) != 1 or not valid_window(row["relevant_windows"][0], row["duration"]):
            raise ValueError(f"Invalid temporal window in {row['qid']}")
        if record["verification_status"] != "ACCEPTED_VISUAL_REVIEW" or record["reviewer_agreement_tiou"] < 0.70:
            raise ValueError(f"Invalid verification status on {row['qid']}")
        for stage in ("generation", "review"):
            events = record.get(stage, {}).get("events", [])
            if not events or events[-1].get("decision") != "accept":
                raise ValueError(f"Missing accept decision in {stage} for {row['qid']}")

    features, diversity_report = release_quality(parent, work_dir, new_rows, similarity=0.94)
    new_features = {str(r["qid"]): features["text"][str(r["qid"])] for r in new_rows}
    for qid, feat in new_features.items():
        arr = feat.array()
        if arr.ndim != 2 or arr.shape[1] != 512:
            raise ValueError(f"Unexpected text tensor shape for {qid}: {arr.shape}")
    quality_report = {
        "validated_text_features": len(new_features),
        "feature_errors": {},
        "diversity": diversity_report,
    }
    if diversity_report["exact_duplicates"] or diversity_report["flags"]:
        raise ValueError(f"Diversity check failed: {diversity_report}")

    orig_test_bytes = (parent.frozen / "test.jsonl").read_bytes()
    new_test_bytes = jsonl_bytes(new_rows)
    test300_bytes = orig_test_bytes + (b"" if orig_test_bytes.endswith(b"\n") else b"\n") + new_test_bytes
    combined_rows = orig_test_rows + new_rows
    if len(combined_rows) != 300 or len({r["vid"] for r in combined_rows}) != 150:
        raise ValueError("Combined Test-300 must contain 300 queries across 150 unique test videos")

    queries_per_video = Counter(r["vid"] for r in combined_rows)
    multiplicity_hist = dict(sorted(Counter(queries_per_video.values()).items()))

    return {
        "audit": audit_report,
        "new_rows": new_rows,
        "sidecar": sidecar,
        "progress": {k: v for k, v in prog.items() if k != "videos"},
        "quality": quality_report,
        "new_features": new_features,
        "orig_test_bytes": orig_test_bytes,
        "new_test_bytes": new_test_bytes,
        "test300_bytes": test300_bytes,
        "combined_rows": combined_rows,
        "multiplicity_hist": {str(k): v for k, v in multiplicity_hist.items()},
        "duration_buckets": {
            "original_test150": duration_distribution(orig_test_rows),
            "new_test150": duration_distribution(new_rows),
            "combined_test300": duration_distribution(combined_rows),
        },
    }


def _portable_recipe(parent: Parent, seed: int) -> dict:
    """Return a clean relative-path Moment-DETR recipe (Train 1200 + Val 150 only)."""
    cfg = read_json(runner.CONFIG)
    return {
        "dataset": "lumae_ads_d1500",
        "canonical_sha256": CANONICAL,
        "split_sha256": {split: parent.hashes[f"{split}.jsonl"] for split in ("train", "val")},
        "counts": {split: COUNTS[split] for split in ("train", "val")},
        "upstream_commit": runner.UPSTREAM,
        "m0_checkpoint": "assets/M0_pretrained.ckpt",
        "m0_sha256": M0,
        "seed": seed,
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "gradient_clip": cfg["gradient_clip"],
        "amp": cfg["amp"],
        "lr_step_size": cfg["lr_step_size"],
        "lr_gamma": cfg["lr_gamma"],
        "normalize_video": cfg["normalize_video"],
        "normalize_text": cfg["normalize_text"],
        "max_video_length": cfg["max_video_length"],
        "max_query_length": cfg["max_query_length"],
        "clip_length": cfg["clip_length"],
        "video_key": "features",
        "video_cache_dim": 512,
        "text_key": "last_hidden_state",
        "text_dim": 512,
        "dataloader_workers": 0,
        "deterministic": True,
        "selection_split": "val",
        "selection_primary": "R1@0.5",
        "selection_tiebreak": "mIoU",
        "test_used_for_selection": False,
    }


def _portable_matrix(parent: Parent) -> dict:
    runs = [
        {
            "model": "moment_detr",
            "seed": seed,
            "execution": "NOT_RUN",
            "selection_split": "val",
            "checkpoint": "assets/M0_pretrained.ckpt",
            "checkpoint_sha256": M0,
            "planned_checkpoint": f"runs/moment_detr/seed_{seed}/best.ckpt",
            "feature_recipe": "Frozen CLIP ViT-B/32 token states and 2-second RGB clips; TEF added on load",
            "training_config": _portable_recipe(parent, seed),
        }
        for seed in SEEDS
    ]
    blocked = {
        name: {"status": "BLOCKED", **details}
        for name, details in BLOCKED.items()
    }
    return {
        "benchmark_version": VERSION,
        "parent_dataset_version": "lumae_ads_d1500",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "runs": runs,
        "blocked": blocked,
    }


def _colab_prepare_script() -> str:
    return r'''"""Prepare unified Google Colab workspace for LUMAE D1500 Test-300 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_zip(src: pathlib.Path, dst: pathlib.Path, prefix: str = "") -> int:
    count = 0
    with zipfile.ZipFile(src, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if prefix and not info.filename.startswith(prefix):
                continue
            target = dst / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info.filename))
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage LUMAE Test-300 features and run preflight.")
    parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--d500-drive-dir", type=pathlib.Path, default=None,
                        help="Folder holding m1_d200_features and m1_d500_features from Drive.")
    parser.add_argument("--d1000-zip", type=pathlib.Path, default=None)
    parser.add_argument("--d1500-zip", type=pathlib.Path, default=None)
    parser.add_argument("--m0-ckpt", type=pathlib.Path, default=None)
    parser.add_argument("--verify-all-hashes", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    ws = args.workspace.resolve()
    sys.path.insert(0, str(ws))
    from experiments.M1_lumae_ads import cached_runner as runner

    if args.d500_drive_dir:
        artifacts_dir = ws / "features" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name in ("m1_d200_features", "m1_d500_features"):
            src = args.d500_drive_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Missing D500 folder: {src}")
            dst = artifacts_dir / name
            if not dst.exists():
                shutil.copytree(src, dst)
    if args.d1000_zip:
        extract_zip(args.d1000_zip, ws, prefix="features/")
    if args.d1500_zip:
        extract_zip(args.d1500_zip, ws, prefix="features/")
    if args.m0_ckpt:
        target_m0 = ws / "assets" / "M0_pretrained.ckpt"
        target_m0.parent.mkdir(parents=True, exist_ok=True)
        if args.m0_ckpt.resolve() != target_m0.resolve():
            shutil.copy2(args.m0_ckpt, target_m0)

    base_index = json.loads((ws / "manifests" / "base_feature_index.json").read_text(encoding="utf-8"))
    new_hashes = json.loads((ws / "manifests" / "test300_text_checksums.json").read_text(encoding="utf-8"))
    feat_manifest = json.loads((ws / "manifests" / "test300_feature_manifest.json").read_text(encoding="utf-8"))

    missing = []
    hash_errors = []
    for kind in ("video", "text"):
        for ident, rel in feat_manifest[kind].items():
            fp = ws / "features" / rel
            if not fp.exists():
                missing.append(f"{kind}:{ident}:{rel}")
                continue
            if args.verify_all_hashes:
                expected = base_index[kind].get(ident, {}).get("sha256") or new_hashes.get(ident)
                if expected and sha256_file(fp) != expected:
                    hash_errors.append(f"{kind}:{ident}")

    m0_ref = json.loads((ws / "manifests" / "m0_reference.json").read_text(encoding="utf-8"))
    m0_path = ws / m0_ref["relative_path"]
    m0_ok = m0_path.exists() and sha256_file(m0_path) == m0_ref["sha256"]

    summary = {
        "missing_features": len(missing),
        "missing_sample": missing[:10],
        "hash_errors": hash_errors,
        "m0_present_and_verified": m0_ok,
    }
    if not args.skip_preflight:
        if missing or hash_errors or not m0_ok:
            print(json.dumps(summary, indent=2))
            raise SystemExit(1)
        preflight_out = ws / "runs" / "preflight_test300.json"
        preflight_out.parent.mkdir(parents=True, exist_ok=True)
        recipe = json.loads((ws / "configs" / "moment_detr_d1500_recipe.json").read_text(encoding="utf-8"))
        datasets, identity, exceptions = runner.preflight(
            train_jsonl=ws / "dataset" / "train.jsonl",
            val_jsonl=ws / "dataset" / "val.jsonl",
            feature_manifest=ws / "manifests" / "test300_feature_manifest.json",
            feature_root=ws / "features",
            upstream_root=ws / "upstream",
            checkpoint=m0_path,
            config=recipe,
        )
        preflight_payload = {"status": "PASS", "identity": identity, "dtype_exceptions": exceptions}
        preflight_out.write_text(json.dumps(preflight_payload, indent=2) + "\n", encoding="utf-8")
        summary["preflight"] = {
            "status": "PASS",
            "train_samples": len(datasets["train"]),
            "val_samples": len(datasets["val"]),
            "verified_tensors": len(identity["feature_sha256"]),
            "dtype_exceptions": exceptions,
            "m0_sha256": identity["m0_sha256"],
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _colab_train_script() -> str:
    return '''"""Train Moment-DETR from M0 on frozen D1500 Train1200 + Val150 (no Test leakage)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Moment-DETR on D1500 Train1200/Val150.")
    parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for smoke tests.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ws = args.workspace.resolve()
    sys.path.insert(0, str(ws))
    from experiments.M1_lumae_ads import cached_runner as runner

    cfg = json.loads((ws / "configs" / "moment_detr_d1500_recipe.json").read_text(encoding="utf-8"))
    cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    m0_path = ws / "assets" / "M0_pretrained.ckpt"
    datasets, identity, _ = runner.preflight(
        train_jsonl=ws / "dataset" / "train.jsonl",
        val_jsonl=ws / "dataset" / "val.jsonl",
        feature_manifest=ws / "manifests" / "test300_feature_manifest.json",
        feature_root=ws / "features",
        upstream_root=ws / "upstream",
        checkpoint=m0_path,
        config=cfg,
    )
    run_dir = ws / "runs" / "moment_detr" / f"seed_{args.seed}"
    history = runner.train(
        datasets=datasets,
        identity=identity,
        upstream_root=ws / "upstream",
        checkpoint=m0_path,
        output=run_dir,
        resume=args.resume,
    )
    print(json.dumps({"seed": args.seed, "epochs_completed": len(history), "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _colab_eval_script() -> str:
    return r'''"""Evaluate Moment-DETR checkpoints across all LUMAE D1500 Test-300 benchmark slices."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import pathlib
import random
import statistics
import sys
import time


def compute_tiou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1] - a[0], 0.0) + max(b[1] - b[0], 0.0) - inter
    return inter / union if union > 0 else 0.0


def evaluate_slice(rows: list[dict], preds: dict[str, list], upstream_root: pathlib.Path,
                   *, bootstrap: int = 1000, seed: int = 20261114, k: int = 5) -> dict:
    import numpy as np
    from standalone_eval.eval import compute_average_precision_detection_wrapper

    thresholds = np.linspace(0.5, 0.95, 10).round(2)
    grouped = defaultdict(list)
    for row in rows:
        qid = str(row["qid"])
        windows = preds[qid]
        target = row["relevant_windows"][0]
        overlap = [compute_tiou(w[:2], target) for w in windows]
        top = overlap[0] if overlap else 0.0
        ranked = [{"video-id": qid, "t-start": s, "t-end": e, "score": sc} for s, e, sc in windows[:10]]
        ref = [{"video-id": qid, "t-start": target[0], "t-end": target[1]}]
        _, ap = compute_average_precision_detection_wrapper((qid, ref, ranked), thresholds)
        grouped[row["vid"]].append((top, max(overlap[:k], default=0.0), float(ap.mean()), target[1] - target[0] <= 2.0))

    def mean_pct(vals: list[float]) -> float | None:
        return round(statistics.mean(vals) * 100, 2) if vals else None

    def summarize(items: list[tuple]) -> dict:
        return {
            "R1@0.5": mean_pct([int(t >= 0.5) for t, _, _, _ in items]),
            "R1@0.7": mean_pct([int(t >= 0.7) for t, _, _, _ in items]),
            "mIoU": mean_pct([t for t, _, _, _ in items]),
            "MR_mAP": mean_pct([ap for _, _, ap, _ in items]),
            f"Oracle@{k}/0.5": mean_pct([int(b >= 0.5) for _, b, _, _ in items]),
            "short_R1@0.5": mean_pct([int(t >= 0.5) for t, _, _, s in items if s]),
        }

    result = summarize([item for g in grouped.values() for item in g])
    if bootstrap > 0:
        rng = random.Random(seed)
        groups = list(grouped.values())
        draws = [summarize([item for g in rng.choices(groups, k=len(groups)) for item in g]) for _ in range(bootstrap)]
        ci95 = {}
        for metric_name in result:
            vals = [d[metric_name] for d in draws if d[metric_name] is not None]
            ci95[metric_name] = np.percentile(vals, [2.5, 97.5]).round(2).tolist() if vals else None
        result["video_bootstrap"] = {"seed": seed, "draws": bootstrap, "ci95": ci95}
    return {"query_count": len(rows), "video_count": len(grouped), **result}


def predict_split(ws: pathlib.Path, annotations: pathlib.Path, checkpoint: pathlib.Path | None,
                  device: str) -> tuple[dict[str, list], dict]:
    import torch
    from torch.utils.data import DataLoader
    from experiments.M1_lumae_ads import cached_runner as runner

    upstream_root = ws / "upstream"
    runner.bind_upstream(upstream_root)
    from moment_detr.model import build_model
    from moment_detr.span_utils import span_cxw_to_xx
    from moment_detr.start_end_dataset import prepare_batch_inputs, start_end_collate

    manifest = json.loads((ws / "manifests" / "test300_feature_manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((ws / "configs" / "moment_detr_d1500_recipe.json").read_text(encoding="utf-8"))
    ds = runner.dataset_type(upstream_root)(annotations, ws / "features", manifest, cfg)

    ckpt_path = checkpoint or (ws / "assets" / "M0_pretrained.ckpt")
    if checkpoint is None or ckpt_path.name == "M0_pretrained.ckpt":
        model, _, _ = runner.load_model(upstream_root, ckpt_path, device)
    else:
        with torch.serialization.safe_globals([argparse.Namespace]):
            saved = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        opt = saved["opt"]
        opt.device = torch.device(device)
        model, _ = build_model(opt)
        model.load_state_dict(saved["model"], strict=True)
        model.to(device)
    model.eval()

    preds: dict[str, list] = {}
    elapsed = 0.0
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=start_end_collate)
    with torch.inference_mode():
        for meta, data in loader:
            inputs, _ = prepare_batch_inputs(data, torch.device(device))
            start = time.perf_counter()
            out = model(**inputs)
            elapsed += time.perf_counter() - start
            windows = span_cxw_to_xx(out["pred_spans"]).cpu().tolist()
            scores = out["pred_logits"].softmax(-1)[..., 0].cpu().tolist()
            for row, spans, confs in zip(meta, windows, scores):
                dur = float(row["duration"])
                ranked = sorted(zip(spans, confs), key=lambda pair: -pair[1])
                clipped = [
                    [max(0.0, a * dur), min(dur, b * dur), float(sc)]
                    for (a, b), sc in ranked
                    if 0.0 <= max(0.0, a * dur) < min(dur, b * dur) <= dur and math.isfinite(sc)
                ]
                preds[str(row["qid"])] = sorted(clipped, key=lambda w: -w[2])
    return preds, {"device": device, "forward_ms_per_query": elapsed * 1000.0 / max(1, len(preds))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on LUMAE Test-300 slices.")
    parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--split", choices=["val", "test300"], default="test300")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    ws = args.workspace.resolve()
    sys.path.insert(0, str(ws))
    from experiments.M1_lumae_ads import cached_runner as runner

    runner.bind_upstream(ws / "upstream")
    if args.split == "val":
        ann = ws / "dataset" / "val.jsonl"
        slices = {"val150": [json.loads(l) for l in ann.read_text(encoding="utf-8").splitlines() if l.strip()]}
    else:
        ann = ws / "dataset" / "test300.jsonl"
        slices = {
            name: [json.loads(l) for l in (ws / "dataset" / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
            for name, fname in (
                ("legacy_test30", "legacy_test30.jsonl"),
                ("other_test120", "new_test120.jsonl"),
                ("original_test150", "original_test150.jsonl"),
                ("new_test150", "new_test150.jsonl"),
                ("combined_test300", "test300.jsonl"),
            )
        }

    preds, timing = predict_split(ws, ann, args.checkpoint, args.device)
    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint or "assets/M0_pretrained.ckpt"),
        "timing": timing,
        "slices": {
            name: evaluate_slice(rows, preds, ws / "upstream", bootstrap=args.bootstrap)
            for name, rows in slices.items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _freeze_report_markdown(state: dict, release_manifest: dict) -> str:
    p = state["progress"]
    q = state["quality"]
    return f"""# LUMAE D1500 Test-300 Dataset Freeze Report

- **Release Version**: `{VERSION}`
- **Parent Dataset**: `lumae_ads_d1500` (canonical SHA-256 `{CANONICAL}`)
- **Parent Split Freeze**: `train=1200`, `val=150`, `test=150` (`legacy_test30=30`, `new_test120=120`) — unchanged
- **Decision**: `FROZEN_INTERIM_BENCHMARK_READY_FOR_COLAB`

## 1. Query & Video Counts

| Slice | Queries | Unique Videos | Annotation Provenance |
| --- | ---: | ---: | --- |
| Original `legacy_test30` | 30 | 30 | Original D1500 Test slice (`AI_PSEUDO_LABELED`) |
| Original `new_test120` | 120 | 120 | Original D1500 Test slice (`AI_PSEUDO_LABELED`) |
| **Original `test150`** | **150** | **150** | Original D1500 frozen Test split |
| **New `new_test150`** | **150** | **75** | `AI_PSEUDO_VERIFIED_BY_AI` (2 new queries per video) |
| **Combined `test300`** | **300** | **150** | `150` original + `150` AI-reviewed (`75` videos x 3 queries, `75` videos x 1 query) |

Full frozen D1500 + Test-300 totals: `1,200` Train + `150` Val + `300` Test = `1,650` temporal grounding queries across `1,500` unique advertisement videos.

## 2. Quality & Blind AI Review Ledger

- **Generation Decisions**: `{json.dumps(p["generation_decisions"])}`
- **Blind Review Decisions**: `{json.dumps(p["review_decisions"])}`
- **Attempt Outcomes Across All 150 Videos**: `{json.dumps(p["attempt_outcomes"])}`
- **Retry Attempts**: `{p["retry_attempts"]}`
- **Accepted Agreement Threshold**: `tIoU >= 0.70`
- **Mean Reviewer Agreement tIoU (150 accepted queries)**: `{p["mean_agreement_tiou"]:.6f}`
- **Mean Generation Confidence (150 accepted queries)**: `{p["mean_generation_confidence"]:.6f}`
- **Mean Review Confidence (150 accepted queries)**: `{p["mean_review_confidence"]:.6f}`
- **Validated New CLIP Text Embeddings**: `{q["validated_text_features"]} / 150` (CLIP ViT-B/32 `openai/clip-vit-base-patch32`, `feature_errors={len(q["feature_errors"])}`)
- **Cosine Diversity Check (`>= 0.94`)**: `exact_duplicates=0`, `flags=0`

## 3. Duration Bucket Distribution

- `original_test150`: `{json.dumps(state["duration_buckets"]["original_test150"])}`
- `new_test150`: `{json.dumps(state["duration_buckets"]["new_test150"])}`
- `combined_test300`: `{json.dumps(state["duration_buckets"]["combined_test300"])}`

## 4. Known Limitations

1. **AI Pseudo-Label Provenance**: All 150 newly added Test-300 queries are AI-generated (`Codex GPT-6`) and blindly AI-reviewed (`GPT-6` on the first 55 videos, `Gemini 4` isolated blind reviewer subagents on the remaining 20 videos). No human ground-truth verification was performed on these 150 new queries.
2. **Incomplete 3-Query Density**: Only 75 of the 150 Test videos have 3 queries (1 original + 2 new); the remaining 75 Test videos currently retain 1 original query. Therefore `combined_test300` mixes query multiplicities (`75` x 3 queries + `75` x 1 query) and must be reported alongside `original_test150` (`legacy_test30`, `other_test120`) and `new_test150`, with video-clustered bootstrap resampling.
3. **Model Baselines**: Only Moment-DETR has a verified cached-feature pipeline (`CLIP ViT-B/32` 512-d + 2-d TEF). `qd_detr` and `univtg` remain `BLOCKED` due to unverified candidate checkpoints and missing aligned SlowFast 2304-d streams.
4. **Test-450 Ledger Preserved Open**: `research_outputs/d1500_test450_ai_v1/release/manifest.json` remains uncreated so expansion from 300 to 450 queries can resume cleanly later.

## 5. Key Release File SHA-256 Checksums

- `test300.jsonl`: `{release_manifest["files"]["test300.jsonl"]}`
- `new_test150.jsonl`: `{release_manifest["files"]["new_test150.jsonl"]}`
- `original_test150.jsonl`: `{release_manifest["files"]["original_test150.jsonl"]}`
- `provenance_test150.jsonl`: `{release_manifest["files"]["provenance_test150.jsonl"]}`
"""


def _colab_runbook_markdown() -> str:
    return f"""# LUMAE Test-300 Google Colab Runbook

This runbook stages `LUMAE_D1500_TEST300_AI_V1` in Google Colab from manually uploaded Google Drive artifacts without modifying any dataset splits or leaking Test queries into training selection.

## 1. Required Drive Artifacts

Upload **only two new ZIPs** into `{DRIVE_ARTIFACT_ROOT}/packages/`:
1. `lumae_test300_incremental.zip` (Package A — Test-300 annotations, provenance, and 150 new CLIP text NPZs)
2. `lumae_colab_benchmark.zip` (Package B — pinned upstream Moment-DETR code, runner, frozen Train/Val/Test splits, base feature index, configs, scripts)

Reuse existing Drive / prior ZIP assets (Package C is **not required**):
- Existing Drive D500 historical folders (`m1_d200_features`, `m1_d500_features`): {DRIVE_LINKS["d500_historical_features_folder"]}
- Verified M0 checkpoint (`M0_pretrained.ckpt`, SHA-256 `{M0}`): {DRIVE_LINKS["m0_pretrained_checkpoint_file"]}
- Existing D1000 cached ZIP (`lumae_d1000_colab.zip`)
- Existing D1500 incremental cached ZIP (`lumae_d1500_incremental_colab.zip`)

## 2. Extract & Install Dependencies in Colab

```bash
mkdir -p /content/lumae_work
cd /content/lumae_work

# Install Python dependencies
pip install -q torch numpy scipy pandas tqdm

# Extract Package B (benchmark code, configs, Train/Val/Test frozen splits)
unzip -q "/content/drive/MyDrive/{DRIVE_ARTIFACT_ROOT}/packages/lumae_colab_benchmark.zip" -d .

# Extract Package A (Test-300 JSONLs, provenance, and 150 new CLIP text NPZs)
unzip -q -o "/content/drive/MyDrive/{DRIVE_ARTIFACT_ROOT}/packages/lumae_test300_incremental.zip" -d .
```

## 3. Stage All 3,150 NPZs & Run Preflight

```bash
python scripts/colab_prepare_workspace.py \\
  --workspace . \\
  --d500-drive-dir "/content/drive/MyDrive/CV_Lumae/artifacts" \\
  --d1000-zip "/content/drive/MyDrive/CV_Lumae/artifacts/lumae_d1000_colab.zip" \\
  --d1500-zip "/content/drive/MyDrive/CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1500_incremental_colab.zip" \\
  --m0-ckpt "/content/drive/MyDrive/CV_Lumae/assets/M0_pretrained.ckpt" \\
  --verify-all-hashes
```

## 4. Evaluate M0 Zero-Shot on Val & Test-300

```bash
python scripts/colab_eval_test300.py --workspace . --split val --device cuda --output runs/m0_eval_val150.json
python scripts/colab_eval_test300.py --workspace . --split test300 --device cuda --output runs/m0_eval_test300.json
```

## 5. Train Moment-DETR (3 Seeds on Train1200 + Val150 Only)

Note: `cached_runner.py` enables `torch.use_deterministic_algorithms(True)` and logs `git rev-parse HEAD`, so set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and ensure the workspace has an initialized git repo (`git init && git commit --allow-empty -m colab` if unzipped fresh without `.git`).

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[ -d .git ] || (git init && git -c user.name=lumae -c user.email=lumae@local commit --allow-empty -m "colab workspace")

for SEED in 42 123 2026; do
  python scripts/colab_train_moment_detr.py --workspace . --seed $SEED
  python scripts/colab_eval_test300.py \\
    --workspace . \\
    --split test300 \\
    --checkpoint "runs/moment_detr/seed_${{SEED}}/best.ckpt" \\
    --device cuda \\
    --output "runs/moment_detr/seed_${{SEED}}/eval_test300.json"
done
```

## 6. Protocol Guardrails

- Select checkpoints strictly by `Val150` (`R1@0.5`, tie-break `mIoU`). Never use Test-300 for hyperparameter tuning or early stopping.
- Always report `legacy_test30`, `other_test120`, `original_test150`, `new_test150`, and `combined_test300` separately alongside video-clustered 95% bootstrap CIs.
- Do NOT execute `qd_detr` or `univtg` (`BLOCKED`).
"""


def _write_clean_zip(zip_path: pathlib.Path, entries: list[tuple[str, bytes]]) -> dict:
    """Create a deterministic ZIP file using fixed timestamps and posix relative paths."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    sizes = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, data in sorted(entries, key=lambda item: item[0]):
            if "\\" in arcname or ":" in arcname or arcname.startswith("/"):
                raise ValueError(f"Non-portable zip entry name: {arcname}")
            if arcname in seen:
                raise ValueError(f"Duplicate zip entry: {arcname}")
            seen.add(arcname)
            info = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_DT)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            sizes[arcname] = len(data)
    return {"file_count": len(entries), "uncompressed_bytes": sum(sizes.values()), "compressed_bytes": zip_path.stat().st_size}


FROZEN_IMMUTABLE_PREFIXES = (
    "dataset/train.jsonl",
    "dataset/val.jsonl",
    "dataset/test.jsonl",
    "dataset/legacy_test30.jsonl",
    "dataset/new_test120.jsonl",
    "dataset/original_test150.jsonl",
    "dataset/split_manifest.json",
    "release/legacy_test30.jsonl",
    "release/new_test120.jsonl",
    "release/original_test150.jsonl",
)


def _check_text_portability(rel_name: str, text: str, *, frozen_test_count: int = 150) -> bool:
    """Return True if a newly authored file leaks local machine paths."""
    if rel_name in FROZEN_IMMUTABLE_PREFIXES:
        return False
    if rel_name in {"dataset/test300.jsonl", "release/test300.jsonl"}:
        lines = [line for line in text.splitlines() if line.strip()]
        text = "\n".join(lines[frozen_test_count:])
    return bool(FORBIDDEN_LOCAL_PATH_RE.search(text))


def verify_zip_portable(zip_path: pathlib.Path) -> dict:
    """Verify ZIP readability via testzip() and scan generated members for local paths."""
    bad = []
    files = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt zip member {corrupt} in {zip_path}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            files += 1
            if "\\" in info.filename or ":" in info.filename or info.filename.startswith("/"):
                bad.append(f"name:{info.filename}")
            if info.filename.endswith((".json", ".jsonl", ".md", ".py", ".txt")):
                text = zf.read(info.filename).decode("utf-8")
                if info.filename.endswith(".py"):
                    ast.parse(text, filename=info.filename)
                if _check_text_portability(info.filename, text):
                    bad.append(f"content:{info.filename}")
    if bad:
        raise ValueError(f"Non-portable path references in {zip_path.name}: {bad}")
    return {"zip_path": str(zip_path.name), "file_count": files, "testzip": "OK", "non_portable_refs": 0}


def build_and_package(
    *,
    work_dir: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
) -> dict:
    """Validate Test-300 ledger, freeze artifacts, and build upload-ready Drive/Colab packages."""
    work_dir = work_dir or (ROOT / "research_outputs" / "d1500_test450_ai_v1")
    out_dir = out_dir or (ROOT / "research_outputs" / "d1500_test300_ai_v1")
    parent = Parent()

    state = collect_test300_state(parent, work_dir)
    base_index = build_base_feature_index(parent)

    # Combine base feature paths + 150 new Test-300 text NPZs under unified relative paths.
    new_text_manifest = {}
    new_text_checksums = {}
    new_text_zip_entries = []
    for row in state["new_rows"]:
        qid = str(row["qid"])
        feat = state["new_features"][qid]
        npz_path = text_path(work_dir, row)
        json_path = npz_path.with_suffix(".json")
        rel_npz = f"test300_text_features/{qid}/{npz_path.name}"
        rel_json = f"test300_text_features/{qid}/{json_path.name}"
        new_text_manifest[qid] = rel_npz
        new_text_checksums[qid] = feat.sha256
        new_text_zip_entries.append((f"features/{rel_npz}", npz_path.read_bytes()))
        new_text_zip_entries.append((f"features/{rel_json}", json_path.read_bytes()))

    combined_feature_manifest = {
        "canonical_sha256": CANONICAL,
        "feature_contract": {"tef_in_cache": False},
        "video": {vid: rec["manifest_path"] for vid, rec in base_index["video"].items()},
        "text": {
            **{qid: rec["manifest_path"] for qid, rec in base_index["text"].items()},
            **dict(sorted(new_text_manifest.items())),
        },
    }
    if len(combined_feature_manifest["video"]) != 1500 or len(combined_feature_manifest["text"]) != 1650:
        raise ValueError("Combined Test-300 feature manifest must map 1500 videos and 1650 queries")

    legacy30_bytes = (parent.frozen / "legacy_test30.jsonl").read_bytes()
    new120_bytes = (parent.frozen / "new_test120.jsonl").read_bytes()
    provenance_bytes = jsonl_bytes(state["sidecar"])
    split_checksums = {
        "canonical_sha256": CANONICAL,
        "split_seal_sha256": SEAL,
        "split_sha256": parent.hashes,
        "counts": COUNTS,
    }
    release_files = {
        "dataset/original_test150.jsonl": state["orig_test_bytes"],
        "dataset/legacy_test30.jsonl": legacy30_bytes,
        "dataset/new_test120.jsonl": new120_bytes,
        "dataset/new_test150.jsonl": state["new_test_bytes"],
        "dataset/test300.jsonl": state["test300_bytes"],
        "provenance/provenance_test150.jsonl": provenance_bytes,
    }
    short_file_hashes = {pathlib.PurePosixPath(k).name: sha_bytes(v) for k, v in release_files.items()}
    integrity_report = {
        "version": VERSION,
        "parent_audit": state["audit"],
        "progress": state["progress"],
        "quality": state["quality"],
        "counts": {
            "train_queries": COUNTS["train"],
            "val_queries": COUNTS["val"],
            "original_test_queries": COUNTS["test"],
            "new_test_queries": len(state["new_rows"]),
            "combined_test_queries": len(state["combined_rows"]),
            "combined_unique_test_videos": 150,
            "queries_per_test_video_histogram": state["multiplicity_hist"],
            "total_frozen_queries": 1650,
            "total_frozen_videos": 1500,
        },
        "duration_buckets": state["duration_buckets"],
        "zero_failures_asserted": {
            "duplicate_qids": 0,
            "invalid_temporal_windows": 0,
            "missing_review_records": 0,
            "missing_text_embeddings": 0,
            "feature_errors": len(state["quality"]["feature_errors"]),
            "diversity_flags": len(state["quality"]["diversity"]["flags"]),
            "exact_duplicates": len(state["quality"]["diversity"]["exact_duplicates"]),
        },
    }
    release_manifest = {
        "dataset_version": VERSION,
        "parent_dataset_version": "lumae_ads_d1500",
        "parent_canonical_sha256": CANONICAL,
        "parent_split_sha256": parent.hashes,
        "counts": integrity_report["counts"],
        "agreement_threshold": 0.70,
        "similarity_threshold": 0.94,
        "quality": state["quality"],
        "text_encoder": {
            "weights_sha256": WEIGHTS_HASH,
            "commit": runner.UPSTREAM,
            "context_length": 77,
            "max_valid_tokens": 32,
            "extractor_sha256": EXTRACTOR_HASHES,
        },
        "files": {k: sha_bytes(v) for k, v in release_files.items()},
        "text_features": {
            qid: {
                "path": f"features/{new_text_manifest[qid]}",
                "sha256": new_text_checksums[qid],
            }
            for qid in sorted(new_text_manifest)
        },
    }
    release_manifest_with_short = {**release_manifest, "files": short_file_hashes}
    freeze_report_md = _freeze_report_markdown(state, release_manifest_with_short).encode("utf-8")
    runbook_md = _colab_runbook_markdown().encode("utf-8")
    recipe_json = _portable_recipe(parent, 42)
    matrix_json = _portable_matrix(parent)
    m0_ref = {
        "filename": "M0_pretrained.ckpt",
        "relative_path": "assets/M0_pretrained.ckpt",
        "repo_relative_source": "research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt",
        "sha256": M0,
        "drive_link": DRIVE_LINKS["m0_pretrained_checkpoint_file"],
    }
    requirements_txt = (
        "torch>=2.1.0\nnumpy>=1.24.0\nscipy>=1.10.0\npandas>=2.0.0\ntqdm>=4.66.0\n"
    ).encode("utf-8")

    # Write organized clean local release files under research_outputs/d1500_test300_ai_v1/
    _write_bytes(out_dir / "release" / "original_test150.jsonl", state["orig_test_bytes"])
    _write_bytes(out_dir / "release" / "legacy_test30.jsonl", legacy30_bytes)
    _write_bytes(out_dir / "release" / "new_test120.jsonl", new120_bytes)
    _write_bytes(out_dir / "release" / "new_test150.jsonl", state["new_test_bytes"])
    _write_bytes(out_dir / "release" / "test300.jsonl", state["test300_bytes"])
    _write_bytes(out_dir / "release" / "provenance_test150.jsonl", provenance_bytes)
    _write_json(out_dir / "release" / "parent_split_checksums.json", split_checksums)
    _write_json(out_dir / "release" / "integrity_report.json", integrity_report)
    _write_json(out_dir / "release" / "manifest.json", release_manifest)

    _write_json(out_dir / "manifests" / "base_feature_index.json", base_index)
    _write_json(out_dir / "manifests" / "test300_feature_manifest.json", combined_feature_manifest)
    _write_json(out_dir / "manifests" / "test300_text_checksums.json", dict(sorted(new_text_checksums.items())))
    _write_json(out_dir / "manifests" / "m0_reference.json", m0_ref)

    _write_json(out_dir / "configs" / "moment_detr_d1500_recipe.json", recipe_json)
    _write_json(out_dir / "configs" / "experiment_matrix.json", matrix_json)

    _write_bytes(out_dir / "DATASET_FREEZE_REPORT.md", freeze_report_md)
    _write_bytes(out_dir / "COLAB_RUNBOOK.md", runbook_md)

    # Assemble Package A: lumae_test300_incremental.zip
    pkg_a_entries: list[tuple[str, bytes]] = [
        ("dataset/original_test150.jsonl", state["orig_test_bytes"]),
        ("dataset/legacy_test30.jsonl", legacy30_bytes),
        ("dataset/new_test120.jsonl", new120_bytes),
        ("dataset/new_test150.jsonl", state["new_test_bytes"]),
        ("dataset/test300.jsonl", state["test300_bytes"]),
        ("dataset/parent_split_checksums.json", _json_bytes(split_checksums)),
        ("provenance/provenance_test150.jsonl", provenance_bytes),
        ("manifests/test300_feature_manifest.json", _json_bytes(combined_feature_manifest)),
        ("manifests/test300_text_checksums.json", _json_bytes(dict(sorted(new_text_checksums.items())))),
        ("release/integrity_report.json", _json_bytes(integrity_report)),
        ("release/manifest.json", _json_bytes(release_manifest)),
        ("DATASET_FREEZE_REPORT.md", freeze_report_md),
        *new_text_zip_entries,
    ]
    pkg_a_path = out_dir / "packages" / "lumae_test300_incremental.zip"
    pkg_a_meta = _write_clean_zip(pkg_a_path, pkg_a_entries)
    pkg_a_check = verify_zip_portable(pkg_a_path)

    # Assemble Package B: lumae_colab_benchmark.zip
    pkg_b_entries: list[tuple[str, bytes]] = [
        ("experiments/__init__.py", b""),
        ("experiments/M1_lumae_ads/__init__.py", b""),
        ("experiments/M1_lumae_ads/cached_runner.py", (ROOT / "experiments/M1_lumae_ads/cached_runner.py").read_bytes()),
        ("experiments/M1_lumae_ads/config_d500_cached.json", (ROOT / "experiments/M1_lumae_ads/config_d500_cached.json").read_bytes()),
        ("manifests/base_feature_index.json", _json_bytes(base_index)),
        ("manifests/m0_reference.json", _json_bytes(m0_ref)),
        ("configs/moment_detr_d1500_recipe.json", _json_bytes(recipe_json)),
        ("configs/experiment_matrix.json", _json_bytes(matrix_json)),
        ("scripts/colab_prepare_workspace.py", _colab_prepare_script().encode("utf-8")),
        ("scripts/colab_train_moment_detr.py", _colab_train_script().encode("utf-8")),
        ("scripts/colab_eval_test300.py", _colab_eval_script().encode("utf-8")),
        ("requirements_colab.txt", requirements_txt),
        ("COLAB_RUNBOOK.md", runbook_md),
    ]
    upstream_root = SOURCE / "upstream"
    for p in sorted(upstream_root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            pkg_b_entries.append((f"upstream/{p.relative_to(upstream_root).as_posix()}", p.read_bytes()))

    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "legacy_test30.jsonl", "new_test120.jsonl", "split_manifest.json", "SHA256SUMS.txt"):
        pkg_b_entries.append((f"dataset/{name}", (parent.frozen / name).read_bytes()))

    for sub_pkg in ("schemas", "io", "validation", "metrics"):
        pkg_b_entries.extend(
            (p.relative_to(ROOT).as_posix(), p.read_bytes())
            for p in sorted((ROOT / "data_process" / sub_pkg).glob("*.py"))
        )
    pkg_b_entries.append(("data_process/__init__.py", (ROOT / "data_process/__init__.py").read_bytes()))

    pkg_b_path = out_dir / "packages" / "lumae_colab_benchmark.zip"
    pkg_b_meta = _write_clean_zip(pkg_b_path, pkg_b_entries)
    pkg_b_check = verify_zip_portable(pkg_b_path)

    pkg_a_sha = _sha256_file(pkg_a_path)
    pkg_b_sha = _sha256_file(pkg_b_path)

    drive_manifest = {
        "benchmark_version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "drive_upload_root": DRIVE_ARTIFACT_ROOT,
        "upload_required": [
            {
                "name": "lumae_test300_incremental.zip",
                "local_relative_path": "packages/lumae_test300_incremental.zip",
                "drive_destination": f"{DRIVE_ARTIFACT_ROOT}/packages/lumae_test300_incremental.zip",
                "sha256": pkg_a_sha,
                "size_bytes": pkg_a_meta["compressed_bytes"],
                "file_count": pkg_a_meta["file_count"],
                "role": "Test-300 JSONLs, provenance, 150 validated new CLIP text NPZs, and freeze manifest.",
            },
            {
                "name": "lumae_colab_benchmark.zip",
                "local_relative_path": "packages/lumae_colab_benchmark.zip",
                "drive_destination": f"{DRIVE_ARTIFACT_ROOT}/packages/lumae_colab_benchmark.zip",
                "sha256": pkg_b_sha,
                "size_bytes": pkg_b_meta["compressed_bytes"],
                "file_count": pkg_b_meta["file_count"],
                "role": "Pinned upstream Moment-DETR code, runner, frozen Train/Val/Test splits, base index, configs, and Colab scripts.",
            },
        ],
        "reuse_existing": [
            {
                "name": "D500 historical CLIP features (m1_d200_features, m1_d500_features)",
                "count": 1000,
                "drive_link": DRIVE_LINKS["d500_historical_features_folder"],
                "action": "Reuse directly in Drive; mount and stage via colab_prepare_workspace.py --d500-drive-dir",
            },
            {
                "name": "lumae_d1000_colab.zip",
                "count": 1000,
                "repo_relative_path": "research_outputs/d1000_agent_20261104_01/lumae_d1000_colab.zip",
                "action": "Reuse existing uploaded ZIP (extract features/ via colab_prepare_workspace.py --d1000-zip)",
            },
            {
                "name": "lumae_d1500_incremental_colab.zip",
                "count": 1000,
                "repo_relative_path": "research_outputs/d1500_agent_20261106_01/lumae_d1500_incremental_colab.zip",
                "action": "Reuse existing uploaded ZIP (extract features/ via colab_prepare_workspace.py --d1500-zip)",
            },
            {
                "name": "M0_pretrained.ckpt",
                "sha256": M0,
                "drive_link": DRIVE_LINKS["m0_pretrained_checkpoint_file"],
                "repo_relative_path": "research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt",
                "action": "Reuse verified Drive file; place at assets/M0_pretrained.ckpt",
            },
        ],
        "package_c_status": {
            "created": False,
            "reason": base_index["package_c_rationale"],
            "verified_base_feature_count": base_index["counts"]["total"],
        },
        "zip_checks": [pkg_a_check, pkg_b_check],
    }
    _write_json(out_dir / "DRIVE_UPLOAD_MANIFEST.json", drive_manifest)

    # Write SHA256SUMS.txt covering all packaged deliverables with POSIX relative paths.
    tracked_files = sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "PACKAGE_VALIDATION_REPORT.md"}
    )
    sha_lines = [f"{_sha256_file(out_dir / rel)}  {rel}" for rel in tracked_files]
    _write_bytes(out_dir / "SHA256SUMS.txt", ("\n".join(sha_lines) + "\n").encode("utf-8"))

    validation_md = f"""# LUMAE Test-300 Package Validation Report

- **Release Version**: `{VERSION}`
- **Generated (UTC)**: `{drive_manifest["generated_utc"]}`
- **Package C Status**: Not created (`3,000 / 3,000` base NPZ features already verified across Drive D500, `lumae_d1000_colab.zip`, and `lumae_d1500_incremental_colab.zip`).

## Package Artifacts

| Archive | Files | Size (bytes) | SHA-256 | Integrity (`testzip`) | Non-Portable Path Refs |
| --- | ---: | ---: | --- | --- | ---: |
| `packages/lumae_test300_incremental.zip` | {pkg_a_meta["file_count"]} | {pkg_a_meta["compressed_bytes"]} | `{pkg_a_sha}` | `{pkg_a_check["testzip"]}` | `{pkg_a_check["non_portable_refs"]}` |
| `packages/lumae_colab_benchmark.zip` | {pkg_b_meta["file_count"]} | {pkg_b_meta["compressed_bytes"]} | `{pkg_b_sha}` | `{pkg_b_check["testzip"]}` | `{pkg_b_check["non_portable_refs"]}` |

## Integrity Checks Run

- Frozen parent dataset audit: `PASS` (`train=1200`, `val=150`, `test=150`, canonical SHA-256 verified)
- Test-300 ledger verification: `PASS` (`150` original + `150` new accepted queries across `150` videos; `0` missing reviews; `0` invalid windows)
- Text feature verification: `PASS` (`150 / 150` new CLIP ViT-B/32 `.npz` tensors verified; `0` feature errors; `0` exact duplicates; `0` diversity flags)
- Portability scan: `PASS` (`0` Windows/dev-machine absolute path references inside either ZIP)
"""
    _write_bytes(out_dir / "PACKAGE_VALIDATION_REPORT.md", validation_md.encode("utf-8"))

    # Final check: no non-portable paths inside generated text files in out_dir either.
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".txt"}:
            rel = p.relative_to(out_dir).as_posix()
            if _check_text_portability(rel, p.read_text(encoding="utf-8")):
                raise ValueError(f"Non-portable local path leaked into {p}")

    return {
        "version": VERSION,
        "output_dir": str(out_dir),
        "packages": {
            "lumae_test300_incremental.zip": {**pkg_a_meta, "sha256": pkg_a_sha, **pkg_a_check},
            "lumae_colab_benchmark.zip": {**pkg_b_meta, "sha256": pkg_b_sha, **pkg_b_check},
        },
        "counts": integrity_report["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate LUMAE Test-300 freeze and Colab packages.")
    parser.add_argument("--work-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    result = build_and_package(work_dir=args.work_dir, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
