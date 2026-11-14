"""Freeze and package the LUMAE D1500 Test-378 benchmark for Drive and Colab."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import pathlib
import statistics
import subprocess
import sys
import zipfile

from experiments.M1_lumae_ads import cached_runner as runner

from .annotation import AnnotationStore, normalized, progress as compute_progress
from .parent import (
    CANONICAL,
    COUNTS,
    INVENTORY,
    M0,
    ROOT,
    SEAL,
    Parent,
    duration_distribution,
    jsonl_bytes,
    read_json,
    sha_bytes,
    valid_window,
)
from .release import EXTRACTOR, EXTRACTOR_HASHES, WEIGHTS_HASH, quality as release_quality, text_path
from .test300 import (
    DRIVE_LINKS,
    FORBIDDEN_LOCAL_PATH_RE,
    FROZEN_IMMUTABLE_PREFIXES,
    _colab_prepare_script,
    _colab_train_script,
    _json_bytes,
    _portable_recipe,
    _sha256_file,
    _write_bytes,
    _write_clean_zip,
    _write_json,
    build_base_feature_index,
)
from .training import BLOCKED, MODELS, SEEDS

VERSION = "LUMAE_D1500_TEST378_AI_V1"
FEATURE_VERSION = "CLIP_VIT_B32_B7E553AC"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1mD7kXYPBf6Xu_IZe6_GpDRju2tKX_-qP"
DRIVE_ARTIFACT_ROOT = "CV_Lumae/artifacts/d1500_test378_ai_v1"
BLOCKED_CLIP_TOKEN_VID = "lumae_ads_a635b3ea335af1fca10b9755f3df72a7"
TARGET_NEW_QUERIES = 228
TARGET_COMPLETED_VIDS = 114
TARGET_TOTAL_TEST_QUERIES = 378
TARGET_TOTAL_CORPUS_QUERIES = 1728

PROVISIONAL_DRIVE_ASSETS = {
    "d1000_colab_zip": {
        "drive_path": "CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1000_colab.zip",
        "local_repo_path": "research_outputs/d1000_agent_20261104_01/lumae_d1000_colab.zip",
        "note": "Observed in user context on Drive; absent from 2026-11-10 scoped listing; live state UNVERIFIED.",
    },
    "d1500_incremental_colab_zip": {
        "drive_path": "CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1500_incremental_colab.zip",
        "local_repo_path": "research_outputs/d1500_agent_20261106_01/lumae_d1500_incremental_colab.zip",
        "note": "Observed in user context on Drive; absent from 2026-11-10 scoped listing; live state UNVERIFIED.",
    },
    "d500_historical_features": {
        "drive_folder_id": "1cTUQP7DVdexC89FJWeWUNoncbBRVhrQo",
        "drive_url": DRIVE_LINKS["d500_historical_features_folder"],
        "drive_subfolders": ["CV_Lumae/artifacts/m1_d200_features", "CV_Lumae/artifacts/m1_d500_features"],
        "expected_npz_count": 1000,
        "logical_bytes": 25902016,
    },
    "m0_pretrained": {
        "drive_id": "1tY354GpxxplRVXQ0yS0Laaus0Am6K-8i",
        "drive_url": DRIVE_LINKS["m0_pretrained_checkpoint_file"],
        "drive_path": "CV_Lumae/checkpoints/moment_detr/pretrained/model_best.ckpt",
        "local_repo_path": "research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt",
        "bytes": 16952721,
        "sha256": M0,
    },
    "d1500_e0_moment_detr": {
        "drive_path": "CV_Lumae/checkpoints/moment_detr/d1500_e0/",
        "local_present": False,
        "note": "Referenced in user context; not materialized locally; live Drive state UNVERIFIED.",
    },
    "qr_detr_candidate": {
        "drive_id": "1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL",
        "drive_url": "https://drive.google.com/file/d/1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL/view?usp=drivesdk",
        "drive_path": "CV_Lumae/checkpoints/QR-DETR/videoonly.ckpt",
        "bytes": 91131233,
        "local_present": False,
    },
    "univtg_candidate_ckpt": {
        "drive_id": "1bRPJkIvUUyaNOKokCtTM_uRzfTZgZ-hc",
        "drive_url": "https://drive.google.com/file/d/1bRPJkIvUUyaNOKokCtTM_uRzfTZgZ-hc/view?usp=drivesdk",
        "drive_path": "CV_Lumae/checkpoints/UniVTG/model_best.ckpt",
        "bytes": 151708747,
        "local_present": False,
    },
    "univtg_candidate_opt": {
        "drive_id": "1_bvni_jCQXodZmZxqBwFtgO3whbnS9OV",
        "drive_url": "https://drive.google.com/file/d/1_bvni_jCQXodZmZxqBwFtgO3whbnS9OV/view?usp=drivesdk",
        "drive_path": "CV_Lumae/checkpoints/UniVTG/opt.json",
        "bytes": 3052,
        "local_present": False,
    },
}


def _git_revision() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _clip_token_counts(rows: list[dict]) -> dict[str, int]:
    """Count BPE content tokens using the pinned CLIP extractor in an isolated check if needed."""
    root = str(EXTRACTOR.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from run_on_video.clip.clip import _tokenizer

    return {str(row["qid"]): len(_tokenizer.encode(row["query"])) for row in rows}


def _summarize_agreement(sidecar: list[dict]) -> dict:
    """Summarize AI generation/review provenance, agreement tIoU, and confidence."""
    tious = [float(rec["reviewer_agreement_tiou"]) for rec in sidecar]
    gen_confs = [float(rec["generation"]["events"][-1]["confidence"]) for rec in sidecar]
    rev_confs = [float(rec["review"]["events"][-1]["confidence"]) for rec in sidecar]
    gen_models = Counter(f"{rec['generation'].get('model')}|{rec['generation'].get('agent')}" for rec in sidecar)
    rev_models = Counter(f"{rec['review'].get('model')}|{rec['review'].get('agent')}" for rec in sidecar)
    tiou_hist = {
        "[0.70,0.80)": sum(0.70 <= x < 0.80 for x in tious),
        "[0.80,0.90)": sum(0.80 <= x < 0.90 for x in tious),
        "[0.90,1.00)": sum(0.90 <= x < 1.00 for x in tious),
        "1.00": sum(x >= 1.0 - 1e-9 for x in tious),
    }
    return {
        "accepted_new_queries": len(sidecar),
        "agreement_threshold": 0.70,
        "tiou_min": round(min(tious), 6),
        "tiou_median": round(statistics.median(tious), 6),
        "tiou_mean": round(statistics.mean(tious), 6),
        "tiou_histogram": tiou_hist,
        "mean_generation_confidence": round(statistics.mean(gen_confs), 6),
        "mean_review_confidence": round(statistics.mean(rev_confs), 6),
        "generation_model_breakdown": dict(sorted(gen_models.items())),
        "review_model_breakdown": dict(sorted(rev_models.items())),
        "human_verified_new_queries": 0,
        "ai_pseudo_label_notice": (
            "All 228 additional Test-378 queries are AI-generated and AI-reviewed pseudo-labels "
            "(zero human verification on added Test queries)."
        ),
    }


def collect_test378_state(parent: Parent, work_dir: pathlib.Path) -> dict:
    """Validate and assemble the deterministic Test-378 snapshot from the persisted ledger."""
    audit_report = parent.audit()
    store = AnnotationStore(parent, work_dir)
    all_rows, all_sidecar, states = store.accepted(0.70)
    token_counts = _clip_token_counts(all_rows)

    selected_pairs = []
    excluded_records = []
    for row, rec in zip(all_rows, all_sidecar):
        if row["vid"] == BLOCKED_CLIP_TOKEN_VID:
            tok = token_counts[str(row["qid"])]
            excluded_records.append({
                "row": row,
                "sidecar": rec,
                "content_tokens": tok,
                "reason": (
                    "Video excluded deterministically at freeze cutoff because its Q1 exceeds "
                    "the pinned CLIP ViT-B/32 30-content-token limit (33 > 30); preserved "
                    "untouched in research_outputs/d1500_test450_ai_v1."
                ),
            })
        else:
            selected_pairs.append((row, rec))

    if len(excluded_records) != 2 or any(
        token_counts[str(r["qid"])] > 30 for r, _ in selected_pairs
    ):
        raise ValueError("Unexpected CLIP token contract state across persisted ledger rows")

    new_rows = [r for r, _ in selected_pairs]
    sidecar = [s for _, s in selected_pairs]

    if len(new_rows) != TARGET_NEW_QUERIES or len(sidecar) != TARGET_NEW_QUERIES:
        raise ValueError(f"Expected {TARGET_NEW_QUERIES} valid added rows, got {len(new_rows)}")

    orig_test_rows = parent.rows["test"]
    orig_test_vids = {r["vid"] for r in orig_test_rows}
    new_vids = {r["vid"] for r in new_rows}
    if len(new_vids) != TARGET_COMPLETED_VIDS or not new_vids.issubset(orig_test_vids):
        raise ValueError("Selected Test-378 added queries must cover exactly 114 frozen Test videos")

    per_vid_new = Counter(r["vid"] for r in new_rows)
    if set(per_vid_new.values()) != {2}:
        raise ValueError("Each completed Test-378 video must have exactly 2 accepted additional queries")

    existing_qids = {str(r["qid"]) for split in ("train", "val", "test") for r in parent.rows[split]}
    new_qids = [str(r["qid"]) for r in new_rows]
    if len(set(new_qids)) != TARGET_NEW_QUERIES or (set(new_qids) & existing_qids):
        raise ValueError("Added Test-378 query IDs must be unique and disjoint from parent splits")

    for split_name in ("train", "val"):
        split_vids = {r["vid"] for r in parent.rows[split_name]}
        if split_vids & orig_test_vids or split_vids & new_vids:
            raise ValueError(f"Test leakage detected against {split_name}")

    for row, record in selected_pairs:
        if len(row["relevant_windows"]) != 1 or not valid_window(row["relevant_windows"][0], row["duration"]):
            raise ValueError(f"Invalid temporal window in {row['qid']}")
        if row.get("saliency_scores") is not None or row.get("relevant_clip_ids") is not None:
            raise ValueError(f"Unexpected saliency field in {row['qid']}")
        if record["verification_status"] != "ACCEPTED_VISUAL_REVIEW" or record["reviewer_agreement_tiou"] < 0.70:
            raise ValueError(f"Invalid review record for {row['qid']}")
        for stage in ("generation", "review"):
            events = record.get(stage, {}).get("events", [])
            if not events or events[-1].get("decision") != "accept":
                raise ValueError(f"Missing accept decision in {stage} for {row['qid']}")

    features, diversity_report = release_quality(parent, work_dir, new_rows, similarity=0.94)
    new_features = {str(r["qid"]): features["text"][str(r["qid"])] for r in new_rows}
    for qid, feat in new_features.items():
        arr = feat.array()
        if arr.ndim != 2 or arr.shape[1] != 512 or arr.shape[0] > 32:
            raise ValueError(f"Unexpected text tensor shape for {qid}: {arr.shape}")
    if diversity_report["exact_duplicates"] or diversity_report["flags"]:
        raise ValueError(f"Diversity check failed: {diversity_report}")

    # Verify normalized query strings have zero exact duplicates even with original Test-150.
    combined_rows = orig_test_rows + new_rows
    if len(combined_rows) != TARGET_TOTAL_TEST_QUERIES or len({r["vid"] for r in combined_rows}) != 150:
        raise ValueError("Combined Test-378 must contain 378 queries across 150 unique Test videos")

    norm_pairs = Counter((r["vid"], normalized(r["query"])) for r in combined_rows)
    if any(cnt > 1 for cnt in norm_pairs.values()):
        raise ValueError("Duplicate normalized (vid, query) found inside Test-378")

    orig_test_bytes = (parent.frozen / "test.jsonl").read_bytes()
    new_test_bytes = jsonl_bytes(new_rows)
    test378_bytes = orig_test_bytes + (b"" if orig_test_bytes.endswith(b"\n") else b"\n") + new_test_bytes

    queries_per_video = Counter(r["vid"] for r in combined_rows)
    multiplicity_hist = dict(sorted(Counter(queries_per_video.values()).items()))
    selected_vid_set = set(new_vids)
    filtered_states = {
        vid: state
        for vid, state in states.items()
        if vid in selected_vid_set or state["status"] != "ACCEPTED_VISUAL_REVIEW"
    }
    prog = compute_progress(new_rows, sidecar, filtered_states)
    agreement = _summarize_agreement(sidecar)
    ledger_snapshot_sha = sha_bytes(
        json.dumps([{"qid": r["qid"], "review_sha": s["review"]["decision_sha256"]} for r, s in selected_pairs], sort_keys=True).encode("utf-8")
    )

    return {
        "audit": audit_report,
        "ledger_snapshot_sha256": ledger_snapshot_sha,
        "all_accepted_rows_count": len(all_rows),
        "new_rows": new_rows,
        "sidecar": sidecar,
        "excluded_records": excluded_records,
        "progress": {k: v for k, v in prog.items() if k != "videos"},
        "agreement": agreement,
        "quality": {
            "validated_text_features": len(new_features),
            "feature_errors": {},
            "diversity": diversity_report,
            "max_clip_content_tokens": max(token_counts[str(r["qid"])] for r in new_rows),
        },
        "new_features": new_features,
        "orig_test_bytes": orig_test_bytes,
        "new_test_bytes": new_test_bytes,
        "test378_bytes": test378_bytes,
        "combined_rows": combined_rows,
        "multiplicity_hist": {str(k): v for k, v in multiplicity_hist.items()},
        "duration_buckets": {
            "original_test150": duration_distribution(orig_test_rows),
            "new_test228": duration_distribution(new_rows),
            "combined_test378": duration_distribution(combined_rows),
        },
    }


def _portable_matrix(parent: Parent) -> dict:
    runs = []
    for model in MODELS:
        for seed in SEEDS:
            if model == "moment_detr":
                runs.append({
                    "model": model,
                    "seed": seed,
                    "status": "READY",
                    "execution": "NOT_RUN",
                    "train_dataset": "lumae_ads_d1500_train1200",
                    "val_dataset": "lumae_ads_d1500_val150",
                    "final_eval_benchmark": VERSION,
                    "initialization_checkpoint": "assets/M0_pretrained.ckpt",
                    "initialization_checkpoint_sha256": M0,
                    "feature_recipe": "Frozen CLIP ViT-B/32 (video 512 + TEF 2 = 514, text 512)",
                    "selection_split": "val",
                    "selection_primary_metric": "mean Val R1@0.5 across 3 seeds",
                    "selection_tiebreak": "Val mIoU",
                    "secondary_metrics": ["R1@0.7", "mIoU", "MR_mAP", "Oracle@5/0.5", "short_R1@0.5"],
                    "planned_checkpoint": f"runs/moment_detr/seed_{seed}/best.ckpt",
                    "output_dir": f"runs/moment_detr/seed_{seed}",
                    "evaluator": "scripts/colab_eval_test378.py",
                    "training_config": _portable_recipe(parent, seed),
                })
            else:
                runs.append({
                    "model": model,
                    "seed": seed,
                    "status": "BLOCKED",
                    "execution": "NOT_RUN",
                    "train_dataset": "lumae_ads_d1500_train1200",
                    "val_dataset": "lumae_ads_d1500_val150",
                    "final_eval_benchmark": VERSION,
                    **BLOCKED[model],
                    "selection_split": "val",
                    "selection_primary_metric": "mean Val R1@0.5 across 3 seeds",
                    "planned_checkpoint": f"runs/{model}/seed_{seed}/best.ckpt",
                    "output_dir": f"runs/{model}/seed_{seed}",
                    "evaluator": "scripts/colab_eval_test378.py",
                    "training_config": None,
                })
    return {
        "benchmark_version": VERSION,
        "parent_dataset_version": "lumae_ads_d1500",
        "split_seed": parent.seed,
        "models": list(MODELS),
        "training_seeds": list(SEEDS),
        "planned_runs": len(runs),
        "selection_protocol": {
            "train_split": "train1200",
            "selection_split": "val150",
            "primary_criterion": "mean_val_R1@0.5",
            "locked_final_eval_split": "test378",
            "never_select_on_test": True,
        },
        "runs": runs,
    }


def _render_test378_script(source: str, orig: str, target: str) -> str:
    return source.replace(orig, target).replace("Test-300", "Test-378").replace("test300", "test378")


def _colab_eval_test378_script() -> str:
    """Adapt the verified Test-300 Colab evaluation script for Test-378 slices."""
    from .test300 import _colab_eval_script

    text = _colab_eval_script()
    text = text.replace("Test-300", "Test-378")
    text = text.replace("test300", "test378")
    text = text.replace('("new_test150", "new_test150.jsonl")', '("new_test228", "new_test228.jsonl")')
    return text


def _colab_audit_candidates_script() -> str:
    return '''"""Colab Phase A audit: inspect Drive checkpoints (M0, QR-DETR, UniVTG) without guessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def inspect_checkpoint(path: pathlib.Path) -> dict:
    import torch

    if not path.is_file():
        return {"present": False, "path": str(path)}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {"present": True, "size_bytes": path.stat().st_size, "sha256": sha}
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        info["top_keys"] = sorted(obj.keys())
        state = obj.get("model") or obj.get("state_dict") or (obj if all(hasattr(v, "shape") for v in obj.values()) else None)
        if isinstance(state, dict):
            info["tensor_count"] = len(state)
            proj_shapes = {k: list(v.shape) for k, v in state.items() if hasattr(v, "shape") and ("input_proj" in k or "vid" in k or "txt" in k or "feat" in k)}
            info["input_projection_shapes"] = dict(list(proj_shapes.items())[:20])
        opt = obj.get("opt")
        if opt is not None:
            opt_dict = vars(opt) if hasattr(opt, "__dict__") else (opt if isinstance(opt, dict) else {})
            for field in ("v_feat_dim", "t_feat_dim", "ctx_mode", "clip_length", "hidden_dim", "num_queries", "span_loss_type", "lw_saliency"):
                if field in opt_dict:
                    info[field] = opt_dict[field]
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local/Drive candidate checkpoints in Colab Phase A.")
    parser.add_argument("--m0-ckpt", type=pathlib.Path, default=pathlib.Path("assets/M0_pretrained.ckpt"))
    parser.add_argument("--qr-detr-ckpt", type=pathlib.Path, default=pathlib.Path("/content/drive/MyDrive/CV_Lumae/checkpoints/QR-DETR/videoonly.ckpt"))
    parser.add_argument("--univtg-ckpt", type=pathlib.Path, default=pathlib.Path("/content/drive/MyDrive/CV_Lumae/checkpoints/UniVTG/model_best.ckpt"))
    parser.add_argument("--univtg-opt", type=pathlib.Path, default=pathlib.Path("/content/drive/MyDrive/CV_Lumae/checkpoints/UniVTG/opt.json"))
    args = parser.parse_args()

    report = {
        "moment_detr_m0": inspect_checkpoint(args.m0_ckpt),
        "qr_detr_candidate": inspect_checkpoint(args.qr_detr_ckpt),
        "univtg_candidate": inspect_checkpoint(args.univtg_ckpt),
        "univtg_opt": json.loads(args.univtg_opt.read_text(encoding="utf-8")) if args.univtg_opt.is_file() else {"present": False},
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _colab_notebook_bytes() -> bytes:
    def md_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}

    def code_cell(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}

    cells = [
        md_cell("# LUMAE D1500 (`LUMAE_D1500_TEST378_AI_V1`) — Colab Phase A & Phase B Benchmark\n\n- **Dataset**: Train `1,200`, Val `150`, Test-378 `378` (`150` original + `228` AI-reviewed queries across `114` x3 and `36` x1 videos)\n- **Protocol**: Train on Train-1200, select checkpoints/architecture strictly on Val-150 (`mean R1@0.5` across seeds `{42, 123, 2026}`), evaluate locked models once on Test-378 with video-clustered bootstrap."),
        code_cell("""from google.colab import drive
drive.mount('/content/drive')
!pip install -q torch numpy scipy pandas tqdm"""),
        code_cell("""%%bash
mkdir -p /content/lumae_work
cd /content/lumae_work
DRIVE_PKG="/content/drive/MyDrive/CV_Lumae/artifacts/d1500_test378_ai_v1/packages"
unzip -q "${DRIVE_PKG}/lumae_colab_benchmark.zip" -d .
unzip -q -o "${DRIVE_PKG}/lumae_test378_incremental.zip" -d .
ls -lh configs manifests dataset scripts"""),
        md_cell("## Phase A.1 — Stage All 3,228 CLIP NPZs & Verify SHA-256 Checksums + Preflight"),
        code_cell("""%%bash
cd /content/lumae_work
python scripts/colab_prepare_workspace.py \\
  --workspace . \\
  --d500-drive-dir "/content/drive/MyDrive/CV_Lumae/artifacts" \\
  --d1000-zip "/content/drive/MyDrive/CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1000_colab.zip" \\
  --d1500-zip "/content/drive/MyDrive/CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1500_incremental_colab.zip" \\
  --m0-ckpt "/content/drive/MyDrive/CV_Lumae/checkpoints/moment_detr/pretrained/model_best.ckpt" \\
  --verify-all-hashes"""),
        md_cell("## Phase A.2 — Audit Candidate Checkpoints (Moment-DETR, QR-DETR, UniVTG)"),
        code_cell("""%%bash
cd /content/lumae_work
python scripts/colab_audit_candidates.py \\
  --m0-ckpt assets/M0_pretrained.ckpt \\
  --qr-detr-ckpt "/content/drive/MyDrive/CV_Lumae/checkpoints/QR-DETR/videoonly.ckpt" \\
  --univtg-ckpt "/content/drive/MyDrive/CV_Lumae/checkpoints/UniVTG/model_best.ckpt" \\
  --univtg-opt "/content/drive/MyDrive/CV_Lumae/checkpoints/UniVTG/opt.json" """),
        md_cell("## Phase A.3 — Verify Evaluator Parity on Val-150 (M0 Zero-Shot)"),
        code_cell("""%%bash
cd /content/lumae_work
python scripts/colab_eval_test378.py --workspace . --split val --device cuda --bootstrap 1000 --output runs/m0_eval_val150.json"""),
        md_cell("## Phase B.1 — Train Moment-DETR Across Seeds `{42, 123, 2026}` (Train-1200 + Val-150 Only)\n\nNote: `qd_detr` and `univtg` are `BLOCKED` until Phase A checkpoint/SlowFast-2304/code-adapter requirements are resolved."),
        code_cell("""%%bash
cd /content/lumae_work
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[ -d .git ] || (git init && git -c user.name=lumae -c user.email=lumae@local commit --allow-empty -m "colab workspace")
for SEED in 42 123 2026; do
  python scripts/colab_train_moment_detr.py --workspace . --seed $SEED
  python scripts/colab_eval_test378.py --workspace . --split val --checkpoint "runs/moment_detr/seed_${SEED}/best.ckpt" --device cuda --output "runs/moment_detr/seed_${SEED}/eval_val150.json"
done"""),
        md_cell("## Phase B.2–B.6 — Locked Final Test-378 Evaluation & Drive Result Export"),
        code_cell("""%%bash
cd /content/lumae_work
python scripts/colab_eval_test378.py --workspace . --split test378 --device cuda --output runs/m0_eval_test378.json
for SEED in 42 123 2026; do
  python scripts/colab_eval_test378.py --workspace . --split test378 --checkpoint "runs/moment_detr/seed_${SEED}/best.ckpt" --device cuda --output "runs/moment_detr/seed_${SEED}/eval_test378.json"
done
mkdir -p /content/drive/MyDrive/CV_Lumae/results/d1500_test378_ai_v1
cp -r runs/* /content/drive/MyDrive/CV_Lumae/results/d1500_test378_ai_v1/"""),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return (json.dumps(nb, indent=2) + "\n").encode("utf-8")


def _freeze_report_md(state: dict, release_manifest: dict, git_rev: str) -> str:
    p = state["progress"]
    ag = state["agreement"]
    q = state["quality"]
    return f"""# LUMAE D1500 Test-378 Dataset Freeze Report

- **Release Version**: `{VERSION}`
- **Feature Version**: `{FEATURE_VERSION}`
- **Repository Commit**: `{git_rev}`
- **Ledger Snapshot SHA-256**: `{state["ledger_snapshot_sha256"]}`
- **Parent Dataset**: `lumae_ads_d1500` (canonical SHA-256 `{CANONICAL}`, split seed `{state["audit"]["split_seed"]}`)
- **Parent Splits**: `train=1200`, `val=150`, `test=150` (`legacy_test30=30`, `new_test120=120`) — unchanged
- **Freeze Status**: `FROZEN_TEST378_BENCHMARK_READY_FOR_COLAB`

## 1. Frozen Dataset Counts

| Slice | Queries | Unique Videos | Annotation Provenance |
| --- | ---: | ---: | --- |
| Train (`train.jsonl`) | 1,200 | 1,200 | `AI_PSEUDO_LABELED` (1,182) + `HUMAN_GT` (18) |
| Validation (`val.jsonl`) | 150 | 150 | `AI_PSEUDO_LABELED` (`30` D500 + `120` D1500) |
| Original `legacy_test30` | 30 | 30 | Original D500 Test (`AI_PSEUDO_LABELED`) |
| Original `other_test120` | 120 | 120 | Original D1500 Test (`AI_PSEUDO_LABELED`) |
| **Original `original_test150`** | **150** | **150** | Frozen D1500 Test split (unchanged) |
| **Added `new_test228`** | **228** | **114** | `AI_PSEUDO_VERIFIED_BY_AI` (`2` new queries per completed video) |
| **Combined `test378`** | **378** | **150** | `150` original + `228` AI-reviewed (`114` videos x 3 queries, `36` videos x 1 query) |

Corpus total: **`1,728` queries across `1,500` unique advertisement videos**.

## 2. Cutoff & Excluded Records Preservation

- Persisted open ledger total at stop (`research_outputs/d1500_test450_ai_v1`): **`{state["all_accepted_rows_count"]}`** accepted queries across `115` completed videos.
- Excluded deterministically: **`2` queries** (`1` video: `{BLOCKED_CLIP_TOKEN_VID}`) because `{BLOCKED_CLIP_TOKEN_VID}_1` tokenizes to `33` CLIP BPE content tokens (`> 30` contract limit enforced by `data_process/test450/release.py`). Neither silent truncation nor query revision/generation after cutoff was performed; the excluded records remain preserved untouched in `research_outputs/d1500_test450_ai_v1` and archived in `release/excluded_test450_records.jsonl`.

## 3. AI Review Agreement & CLIP Feature Validation

- **Generation Decisions**: `{json.dumps(p["generation_decisions"])}`
- **Review Decisions**: `{json.dumps(p["review_decisions"])}`
- **Attempt Outcomes (selected + unresolved)**: `{json.dumps(p["attempt_outcomes"])}`
- **Agreement tIoU (`>= 0.70` gate)**: min `{ag["tiou_min"]:.6f}`, median `{ag["tiou_median"]:.6f}`, mean `{ag["tiou_mean"]:.6f}`
- **tIoU Histogram**: `{json.dumps(ag["tiou_histogram"])}`
- **Mean Generation / Review Confidence**: `{ag["mean_generation_confidence"]:.6f}` / `{ag["mean_review_confidence"]:.6f}`
- **CLIP Text Embeddings Validated**: `{q["validated_text_features"]} / {TARGET_NEW_QUERIES}` (`feature_errors=0`, `max_clip_content_tokens={q["max_clip_content_tokens"]}`)
- **Semantic Similarity & Duplication Check (`cosine >= 0.94`)**: `exact_duplicates=0`, `flags=0`

## 4. Duration Distribution (`seconds`)

- `original_test150`: `{json.dumps(state["duration_buckets"]["original_test150"])}`
- `new_test228`: `{json.dumps(state["duration_buckets"]["new_test228"])}`
- `combined_test378`: `{json.dumps(state["duration_buckets"]["combined_test378"])}`

## 5. Limitations

1. **AI-Only Pseudo-Labels**: All 228 additional Test-378 queries are AI-generated and blindly AI-reviewed (`human_verified_new_queries = 0`). Automated zero-warning validation proves structural and CLIP-contract integrity, NOT human ground-truth accuracy.
2. **Non-Uniform Query Density**: Test-378 has `114` videos with 3 queries and `36` videos with 1 query across `150` unique Test videos (not `378` independent videos). Report slices separately and use video-clustered bootstrap resampling.
3. **Architecture Readiness**: `Moment-DETR` (`514` video w/ TEF + `512` text) is ready for 3-seed Colab execution; `QD-DETR` and `UniVTG` remain `BLOCKED` pending remote checkpoint inspection, model code adapters, and SlowFast 2304-d streams if required.
"""


def _model_compatibility_md() -> str:
    return f"""# LUMAE D1500 Three-Model Compatibility Matrix

Target seeds: `{list(SEEDS)}` | Train split: `Train-1200` | Model selection criterion: `Val-150 mean R1@0.5` | Locked Test: `{VERSION}`

| Model | Local Code | Checkpoint Evidence | Input Feature Contract | Status | Primary Blockers |
| --- | --- | --- | --- | --- | --- |
| **Moment-DETR** | Verified (`upstream/` @ `{runner.UPSTREAM[:12]}` + `experiments/M1_lumae_ads/cached_runner.py`) | Local verified `assets/M0_pretrained.ckpt` (`{M0[:12]}`..., 16.95 MB) + provisional Drive `d1500_e0` (UNVERIFIED remote) | CLIP ViT-B/32 video `[T<=75, 512]` + 2-col TEF on load = `514`, text `[L<=32, 512]`, `lw_saliency=0.0` | **READY** | None |
| **QD-DETR** | Missing locally (`0` files) | Provisional Drive `checkpoints/QR-DETR/videoonly.ckpt` (`91,131,233` B, ID `1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL`, UNVERIFIED locally/remotely) | Stock QVHighlights video-only uses SlowFast `2304` + CLIP `512` + TEF `2` = `2818`; D1500 only caches CLIP `512` | **BLOCKED** | 1. Unverified state dict (`QR-DETR` vs `QD-DETR` projections). 2. Missing aligned SlowFast 2304-d stream (unless CLIP-only checkpoint). 3. Missing QD-DETR code, D1500 adapter & Train-only saliency policy (`saliency_scores=None`). |
| **UniVTG** | Missing locally (`0` files) | Provisional Drive `checkpoints/UniVTG/model_best.ckpt` (`151,708,747` B, ID `1bRPJkIvUUyaNOKokCtTM_uRzfTZgZ-hc`) + `opt.json` (`3,052` B, UNVERIFIED) | Stock UniVTG uses SlowFast `2304` + CLIP `512` = `2816` (`+2 TEF`); D1500 only caches CLIP `512` | **BLOCKED** | 1. Unverified `opt.json` / `model_best.ckpt` compatibility. 2. Missing SlowFast 2304-d stream (if stock config). 3. Missing UniVTG code, `{{qid}}.npz` adapter, seconds conversion & weak-saliency guard (no zero-text fallback). |

Do not assume `E1`/`E2`/`E4` Moment-DETR ablations transfer unchanged to QD-DETR or UniVTG.
"""


def _colab_runbook_md(git_rev: str) -> str:
    return f"""# LUMAE D1500 Test-378 Google Colab Runbook

- **Benchmark**: `{VERSION}` | **Repo Commit**: `{git_rev}`
- **Target Drive Folder**: {DRIVE_FOLDER_URL}

## 1. Upload Packages (`packages/` inside `{DRIVE_ARTIFACT_ROOT}/`)

1. `lumae_test378_incremental.zip` (**Package A** — Test-378 JSONLs, provenance, 228 new CLIP text NPZs, manifests)
2. `lumae_colab_benchmark.zip` (**Package B** — Moment-DETR code, runner, frozen Train/Val/Test splits, scripts, `LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb`)

Conditional Packages C & D are **NOT created** because all `3,000` base NPZs already exist in `m1_d200_features`/`m1_d500_features` (Drive D500) + `lumae_d1000_colab.zip` + `lumae_d1500_incremental_colab.zip`, and `M0_pretrained.ckpt` is provisionally cached on Drive (verify Drive presence manually before Colab).

## 2. Extract in Colab

```bash
mkdir -p /content/lumae_work && cd /content/lumae_work
pip install -q torch numpy scipy pandas tqdm
unzip -q "/content/drive/MyDrive/{DRIVE_ARTIFACT_ROOT}/packages/lumae_colab_benchmark.zip" -d .
unzip -q -o "/content/drive/MyDrive/{DRIVE_ARTIFACT_ROOT}/packages/lumae_test378_incremental.zip" -d .
```

## 3. Stage All 3,228 CLIP NPZs + Run Preflight

```bash
python scripts/colab_prepare_workspace.py \\
  --workspace . \\
  --d500-drive-dir "/content/drive/MyDrive/CV_Lumae/artifacts" \\
  --d1000-zip "/content/drive/MyDrive/CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1000_colab.zip" \\
  --d1500-zip "/content/drive/MyDrive/CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1500_incremental_colab.zip" \\
  --m0-ckpt "/content/drive/MyDrive/CV_Lumae/checkpoints/moment_detr/pretrained/model_best.ckpt" \\
  --verify-all-hashes
```

## 4. Phase A Checkpoint Audit & Val Parity

```bash
python scripts/colab_audit_candidates.py
python scripts/colab_eval_test378.py --workspace . --split val --device cuda --output runs/m0_eval_val150.json
```

## 5. Phase B — 3-Seed Moment-DETR Training & Locked Test-378 Eval

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[ -d .git ] || (git init && git -c user.name=lumae -c user.email=lumae@local commit --allow-empty -m "colab workspace")
for SEED in 42 123 2026; do
  python scripts/colab_train_moment_detr.py --workspace . --seed $SEED
  python scripts/colab_eval_test378.py --workspace . --split val --checkpoint "runs/moment_detr/seed_${{SEED}}/best.ckpt" --device cuda --output "runs/moment_detr/seed_${{SEED}}/eval_val150.json"
done
# After selecting best architecture/config strictly on Val-150:
python scripts/colab_eval_test378.py --workspace . --split test378 --device cuda --output runs/m0_eval_test378.json
for SEED in 42 123 2026; do
  python scripts/colab_eval_test378.py --workspace . --split test378 --checkpoint "runs/moment_detr/seed_${{SEED}}/best.ckpt" --device cuda --output "runs/moment_detr/seed_${{SEED}}/eval_test378.json"
done
```
"""


def verify_test378_zip_portable(zip_path: pathlib.Path) -> dict:
    """Verify ZIP integrity, syntax/JSON validity, and zero leaked local machine paths."""
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
            if info.filename.endswith((".json", ".jsonl", ".md", ".py", ".txt", ".ipynb")):
                text = zf.read(info.filename).decode("utf-8")
                if info.filename.endswith(".py"):
                    ast.parse(text, filename=info.filename)
                elif info.filename.endswith(".ipynb"):
                    nb = json.loads(text)
                    if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
                        bad.append(f"ipynb:{info.filename}")
                if info.filename in FROZEN_IMMUTABLE_PREFIXES:
                    continue
                if info.filename in {"dataset/test378.jsonl", "release/test378.jsonl"}:
                    lines = [line for line in text.splitlines() if line.strip()]
                    text = "\n".join(lines[150:])
                if FORBIDDEN_LOCAL_PATH_RE.search(text):
                    bad.append(f"content:{info.filename}")
    if bad:
        raise ValueError(f"Non-portable path references in {zip_path.name}: {bad}")
    return {"zip_path": str(zip_path.name), "file_count": files, "testzip": "OK", "non_portable_refs": 0}


def build_and_package_test378(
    *,
    work_dir: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
) -> dict:
    """Validate Test-378 cutoff snapshot, freeze artifacts, and build upload-ready packages."""
    work_dir = work_dir or (ROOT / "research_outputs" / "d1500_test450_ai_v1")
    out_dir = out_dir or (ROOT / "research_outputs" / "d1500_test378_ai_v1")
    parent = Parent()
    git_rev = _git_revision()

    state = collect_test378_state(parent, work_dir)
    base_index = build_base_feature_index(parent)

    new_text_manifest = {}
    new_text_checksums = {}
    new_text_zip_entries = []
    for row in state["new_rows"]:
        qid = str(row["qid"])
        feat = state["new_features"][qid]
        npz_path = text_path(work_dir, row)
        json_path = npz_path.with_suffix(".json")
        rel_npz = f"test378_text_features/{qid}/{npz_path.name}"
        rel_json = f"test378_text_features/{qid}/{json_path.name}"
        new_text_manifest[qid] = rel_npz
        new_text_checksums[qid] = feat.sha256
        new_text_zip_entries.append((f"features/{rel_npz}", npz_path.read_bytes()))
        new_text_zip_entries.append((f"features/{rel_json}", json_path.read_bytes()))

    combined_feature_manifest = {
        "canonical_sha256": CANONICAL,
        "feature_version": FEATURE_VERSION,
        "feature_contract": {"tef_in_cache": False, "video_dim_cached": 512, "video_dim_with_tef": 514, "text_dim": 512},
        "video": {vid: rec["manifest_path"] for vid, rec in base_index["video"].items()},
        "text": {
            **{qid: rec["manifest_path"] for qid, rec in base_index["text"].items()},
            **dict(sorted(new_text_manifest.items())),
        },
    }
    if len(combined_feature_manifest["video"]) != 1500 or len(combined_feature_manifest["text"]) != TARGET_TOTAL_CORPUS_QUERIES:
        raise ValueError("Combined Test-378 feature manifest must map 1500 videos and 1728 queries")

    legacy30_bytes = (parent.frozen / "legacy_test30.jsonl").read_bytes()
    new120_bytes = (parent.frozen / "new_test120.jsonl").read_bytes()
    provenance_bytes = jsonl_bytes(state["sidecar"])
    excluded_jsonl_bytes = jsonl_bytes([item["row"] for item in state["excluded_records"]])
    excluded_manifest = {
        "benchmark_version": VERSION,
        "open_ledger_location": "research_outputs/d1500_test450_ai_v1",
        "ledger_total_accepted_queries": state["all_accepted_rows_count"],
        "selected_test378_new_queries": len(state["new_rows"]),
        "excluded_count": len(state["excluded_records"]),
        "selection_rule": (
            "Include all contract-valid accepted videos up to stop cutoff; exclude only "
            "lumae_ads_a635b3ea335af1fca10b9755f3df72a7 because Q1 exceeds 30 CLIP BPE content tokens."
        ),
        "excluded_items": [
            {
                "qid": item["row"]["qid"],
                "vid": item["row"]["vid"],
                "content_tokens": item["content_tokens"],
                "reason": item["reason"],
                "preserved_evidence_ref": item["sidecar"]["evidence_reference"],
            }
            for item in state["excluded_records"]
        ],
    }

    split_checksums = {
        "canonical_sha256": CANONICAL,
        "split_seal_sha256": SEAL,
        "split_sha256": parent.hashes,
        "counts": COUNTS,
    }
    feature_cache_audit = {
        "feature_version": FEATURE_VERSION,
        "extractor_commit": runner.UPSTREAM,
        "clip_weights_sha256": WEIGHTS_HASH,
        "video_entries": 1500,
        "original_text_entries": 1500,
        "added_text_entries": len(new_text_manifest),
        "total_text_entries": len(combined_feature_manifest["text"]),
        "total_npz_entries": 1500 + len(combined_feature_manifest["text"]),
        "dtypes": {
            "video/float32": 1499,
            "video/float16": 1,
            "original_text/float32": 1500,
            "added_text/float32": len(new_text_manifest),
        },
        "legacy_dtype_exception": {
            "vid": "lumae_ads_008ff6f5e15fae24e9a72bc79deae917",
            "stored_dtype": "float16",
            "loader_behavior": "Cast to float32 on load in cached_runner.py",
        },
        "shared_video_multi_query_reuse": True,
        "tef_in_cache": False,
    }

    release_files = {
        "dataset/original_test150.jsonl": state["orig_test_bytes"],
        "dataset/legacy_test30.jsonl": legacy30_bytes,
        "dataset/new_test120.jsonl": new120_bytes,
        "dataset/new_test228.jsonl": state["new_test_bytes"],
        "dataset/test378.jsonl": state["test378_bytes"],
        "provenance/provenance_test228.jsonl": provenance_bytes,
    }
    integrity_report = {
        "version": VERSION,
        "feature_version": FEATURE_VERSION,
        "git_commit": git_rev,
        "ledger_snapshot_sha256": state["ledger_snapshot_sha256"],
        "parent_audit": state["audit"],
        "progress": state["progress"],
        "agreement": state["agreement"],
        "quality": state["quality"],
        "counts": {
            "train_queries": COUNTS["train"],
            "val_queries": COUNTS["val"],
            "original_test_queries": COUNTS["test"],
            "new_test_queries": len(state["new_rows"]),
            "combined_test_queries": len(state["combined_rows"]),
            "combined_unique_test_videos": 150,
            "queries_per_test_video_histogram": state["multiplicity_hist"],
            "total_frozen_queries": TARGET_TOTAL_CORPUS_QUERIES,
            "total_frozen_videos": 1500,
        },
        "duration_buckets": state["duration_buckets"],
        "zero_failures_asserted": {
            "duplicate_qids": 0,
            "invalid_temporal_windows": 0,
            "missing_review_records": 0,
            "missing_text_embeddings": 0,
            "feature_errors": 0,
            "diversity_flags": 0,
            "exact_duplicates": 0,
            "clip_token_overflows_in_selected": 0,
        },
    }
    release_manifest = {
        "dataset_version": VERSION,
        "feature_version": FEATURE_VERSION,
        "parent_dataset_version": "lumae_ads_d1500",
        "parent_canonical_sha256": CANONICAL,
        "parent_split_sha256": parent.hashes,
        "git_commit": git_rev,
        "ledger_snapshot_sha256": state["ledger_snapshot_sha256"],
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
            qid: {"path": f"features/{new_text_manifest[qid]}", "sha256": new_text_checksums[qid]}
            for qid in sorted(new_text_manifest)
        },
    }

    d1000_zip = ROOT / "research_outputs/d1000_agent_20261104_01/lumae_d1000_colab.zip"
    d1500_zip = ROOT / "research_outputs/d1500_agent_20261106_01/lumae_d1500_incremental_colab.zip"
    m0_ckpt = ROOT / "research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt"
    test300_inc = ROOT / "research_outputs/d1500_test300_ai_v1/packages/lumae_test300_incremental.zip"
    test300_bench = ROOT / "research_outputs/d1500_test300_ai_v1/packages/lumae_colab_benchmark.zip"

    local_inventory = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_rev,
        "artifacts": [
            {
                "name": p.name,
                "repo_relative_path": p.relative_to(ROOT).as_posix(),
                "exists": p.is_file(),
                "size_bytes": p.stat().st_size if p.is_file() else None,
                "sha256": _sha256_file(p) if p.is_file() else None,
            }
            for p in (d1000_zip, d1500_zip, m0_ckpt, test300_inc, test300_bench)
        ],
        "base_cached_features": {
            "video_count": 1500,
            "text_count": 1500,
            "missing_on_disk": 0,
            "inventory_sha256": INVENTORY,
        },
        "test378_new_text_features": {
            "count": len(new_text_manifest),
            "bytes": sum(state["new_features"][qid].path.stat().st_size for qid in new_text_manifest),
        },
        "sealed_test300_intact": test300_inc.is_file() and test300_bench.is_file(),
        "missing_local_models": {
            "qd_detr": "Missing local checkpoint, source code, adapter, and aligned SlowFast 2304-d stream.",
            "univtg": "Missing local checkpoint, opt.json, source code, adapter, and SlowFast 2304-d stream.",
            "d1500_e0": "Not materialized locally; referenced provisionally on Drive.",
        },
    }
    drive_inventory = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_drive_folder": DRIVE_FOLDER_URL,
        "remote_access_available": False,
        "remote_verification_status": "UNVERIFIED_NO_DRIVE_AUTH",
        "provisional_evidence_source": "research_outputs/d1500_colab_audit_20261110_01/drive_metadata_snapshot.json + user prompt context",
        "provisional_items": PROVISIONAL_DRIVE_ASSETS,
    }

    freeze_report_md = _freeze_report_md(state, release_manifest, git_rev).encode("utf-8")
    runbook_md = _colab_runbook_md(git_rev).encode("utf-8")
    compat_md = _model_compatibility_md().encode("utf-8")
    recipe_json = _portable_recipe(parent, 42)
    matrix_json = _portable_matrix(parent)
    notebook_bytes = _colab_notebook_bytes()
    m0_ref = {
        "filename": "M0_pretrained.ckpt",
        "relative_path": "assets/M0_pretrained.ckpt",
        "repo_relative_source": "research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt",
        "sha256": M0,
        "size_bytes": 16952721,
        "drive_link": DRIVE_LINKS["m0_pretrained_checkpoint_file"],
    }
    requirements_txt = (
        "torch>=2.1.0\nnumpy>=1.24.0\nscipy>=1.10.0\npandas>=2.0.0\ntqdm>=4.66.0\n"
    ).encode("utf-8")

    # Write local frozen release + manifests + configs + docs under research_outputs/d1500_test378_ai_v1/
    _write_bytes(out_dir / "release" / "original_test150.jsonl", state["orig_test_bytes"])
    _write_bytes(out_dir / "release" / "legacy_test30.jsonl", legacy30_bytes)
    _write_bytes(out_dir / "release" / "new_test120.jsonl", new120_bytes)
    _write_bytes(out_dir / "release" / "new_test228.jsonl", state["new_test_bytes"])
    _write_bytes(out_dir / "release" / "test378.jsonl", state["test378_bytes"])
    _write_bytes(out_dir / "release" / "provenance_test228.jsonl", provenance_bytes)
    _write_bytes(out_dir / "release" / "excluded_test450_records.jsonl", excluded_jsonl_bytes)
    _write_json(out_dir / "release" / "excluded_records_manifest.json", excluded_manifest)
    _write_json(out_dir / "release" / "parent_split_checksums.json", split_checksums)
    _write_json(out_dir / "release" / "review_agreement_report.json", state["agreement"])
    _write_json(out_dir / "release" / "integrity_report.json", integrity_report)
    _write_json(out_dir / "release" / "manifest.json", release_manifest)

    _write_json(out_dir / "manifests" / "base_feature_index.json", base_index)
    _write_json(out_dir / "manifests" / "test378_feature_manifest.json", combined_feature_manifest)
    _write_json(out_dir / "manifests" / "test378_text_checksums.json", dict(sorted(new_text_checksums.items())))
    _write_json(out_dir / "manifests" / "feature_cache_audit.json", feature_cache_audit)
    _write_json(out_dir / "manifests" / "m0_reference.json", m0_ref)

    _write_json(out_dir / "configs" / "moment_detr_d1500_recipe.json", recipe_json)
    _write_json(out_dir / "configs" / "experiment_matrix.json", matrix_json)

    _write_bytes(out_dir / "notebooks" / "LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb", notebook_bytes)
    _write_bytes(out_dir / "DATASET_FREEZE_REPORT.md", freeze_report_md)
    _write_bytes(out_dir / "COLAB_RUNBOOK.md", runbook_md)
    _write_bytes(out_dir / "MODEL_COMPATIBILITY_MATRIX.md", compat_md)
    _write_json(out_dir / "LOCAL_ARTIFACT_INVENTORY.json", local_inventory)
    _write_json(out_dir / "DRIVE_ARTIFACT_INVENTORY.json", drive_inventory)

    # Assemble Package A: lumae_test378_incremental.zip
    pkg_a_entries: list[tuple[str, bytes]] = [
        ("dataset/original_test150.jsonl", state["orig_test_bytes"]),
        ("dataset/legacy_test30.jsonl", legacy30_bytes),
        ("dataset/new_test120.jsonl", new120_bytes),
        ("dataset/new_test228.jsonl", state["new_test_bytes"]),
        ("dataset/test378.jsonl", state["test378_bytes"]),
        ("dataset/parent_split_checksums.json", _json_bytes(split_checksums)),
        ("provenance/provenance_test228.jsonl", provenance_bytes),
        ("provenance/excluded_test450_records.jsonl", excluded_jsonl_bytes),
        ("manifests/test378_feature_manifest.json", _json_bytes(combined_feature_manifest)),
        ("manifests/test378_text_checksums.json", _json_bytes(dict(sorted(new_text_checksums.items())))),
        ("manifests/feature_cache_audit.json", _json_bytes(feature_cache_audit)),
        ("release/review_agreement_report.json", _json_bytes(state["agreement"])),
        ("release/excluded_records_manifest.json", _json_bytes(excluded_manifest)),
        ("release/integrity_report.json", _json_bytes(integrity_report)),
        ("release/manifest.json", _json_bytes(release_manifest)),
        ("DATASET_FREEZE_REPORT.md", freeze_report_md),
        *new_text_zip_entries,
    ]
    pkg_a_path = out_dir / "packages" / "lumae_test378_incremental.zip"
    pkg_a_meta = _write_clean_zip(pkg_a_path, pkg_a_entries)
    pkg_a_check = verify_test378_zip_portable(pkg_a_path)

    # Assemble Package B: lumae_colab_benchmark.zip
    prep_script = _render_test378_script(_colab_prepare_script(), "3,150", "3,228").encode("utf-8")
    train_script = _render_test378_script(_colab_train_script(), "test300", "test378").encode("utf-8")
    eval_script = _colab_eval_test378_script().encode("utf-8")
    audit_script = _colab_audit_candidates_script().encode("utf-8")

    pkg_b_entries: list[tuple[str, bytes]] = [
        ("experiments/__init__.py", b""),
        ("experiments/M1_lumae_ads/__init__.py", b""),
        ("experiments/M1_lumae_ads/cached_runner.py", (ROOT / "experiments/M1_lumae_ads/cached_runner.py").read_bytes()),
        ("experiments/M1_lumae_ads/config_d500_cached.json", (ROOT / "experiments/M1_lumae_ads/config_d500_cached.json").read_bytes()),
        ("manifests/base_feature_index.json", _json_bytes(base_index)),
        ("manifests/feature_cache_audit.json", _json_bytes(feature_cache_audit)),
        ("manifests/m0_reference.json", _json_bytes(m0_ref)),
        ("configs/moment_detr_d1500_recipe.json", _json_bytes(recipe_json)),
        ("configs/experiment_matrix.json", _json_bytes(matrix_json)),
        ("scripts/colab_prepare_workspace.py", prep_script),
        ("scripts/colab_train_moment_detr.py", train_script),
        ("scripts/colab_eval_test378.py", eval_script),
        ("scripts/colab_audit_candidates.py", audit_script),
        ("LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb", notebook_bytes),
        ("requirements_colab.txt", requirements_txt),
        ("COLAB_RUNBOOK.md", runbook_md),
        ("MODEL_COMPATIBILITY_MATRIX.md", compat_md),
        ("DRIVE_ARTIFACT_INVENTORY.json", _json_bytes(drive_inventory)),
    ]
    upstream_root = ROOT / "research_outputs/p0_20261103_01/upstream"
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
    pkg_b_check = verify_test378_zip_portable(pkg_b_path)

    pkg_a_sha = _sha256_file(pkg_a_path)
    pkg_b_sha = _sha256_file(pkg_b_path)

    drive_upload_audit_md = f"""# LUMAE D1500 Google Drive Upload Audit

- **Target Drive Folder**: {DRIVE_FOLDER_URL}
- **Remote Verification Status**: `UNVERIFIED` (no authenticated Drive API / `rclone` / `gdrive` client available in local environment).
- **Basis**: Local SHA-256 verification + `2026-11-10` Drive metadata snapshot (`research_outputs/d1500_colab_audit_20261110_01/`).

| Artifact | Local Status | Drive Status | Decision | Upload Priority |
| --- | --- | --- | --- | --- |
| D500 historical features (`1,000` NPZs) | Verified (100% present) | UNVERIFIED (ID `1cTUQP7DVdexC89FJWeWUNoncbBRVhrQo` in 2026-11-10) | Reuse in Drive | Verify only |
| `lumae_d1000_colab.zip` (`1,000` NPZs) | Verified (`41,623,145` B, `{_sha256_file(d1000_zip)[:12]}`...) | UNVERIFIED (observed in user prompt; absent in 2026-11-10 listing) | Verify on Drive; upload existing ZIP if missing | High (if absent) |
| `lumae_d1500_incremental_colab.zip` (`1,000` NPZs) | Verified (`45,201,054` B, `{_sha256_file(d1500_zip)[:12]}`...) | UNVERIFIED (observed in user prompt; absent in 2026-11-10 listing) | Verify on Drive; upload existing ZIP if missing | High (if absent) |
| Frozen D1500 splits (`train`/`val`/`test`) | Verified (`split_seed=20261110`) | UNVERIFIED (inside `d1500_benchmark_20261110/splits/` + Package B) | Packaged inside Package B | Included in Package B |
| `lumae_test378_incremental.zip` (**Package A**) | **Generated & Verified** (`{pkg_a_meta["compressed_bytes"]}` B, `{pkg_a_sha[:12]}`...) | **MISSING (New)** | **UPLOAD REQUIRED** | **1 (Critical)** |
| `lumae_colab_benchmark.zip` (**Package B**) | **Generated & Verified** (`{pkg_b_meta["compressed_bytes"]}` B, `{pkg_b_sha[:12]}`...) | **MISSING (New)** | **UPLOAD REQUIRED** | **2 (Critical)** |
| Moment-DETR `M0_pretrained.ckpt` | Verified (`16,952,721` B, `{M0[:12]}`...) | UNVERIFIED (ID `1tY354GpxxplRVXQ0yS0Laaus0Am6K-8i` in 2026-11-10) | Reuse in Drive (upload local if absent) | Verify / High if missing |
| Moment-DETR `d1500_e0` | Missing locally | UNVERIFIED (`checkpoints/moment_detr/d1500_e0/`) | Keep on Drive if needed for reference | Low |
| QD-DETR (`QR-DETR/videoonly.ckpt`) | Missing locally (code/SlowFast missing) | UNVERIFIED (ID `1fDFuFXCww_8FeHip-PMNgKj3i1zMHtxL`, 91.13 MB) | `BLOCKED` — inspect in Colab Phase A | Deferred |
| UniVTG (`model_best.ckpt`, `opt.json`) | Missing locally (code/SlowFast missing) | UNVERIFIED (151.71 MB + 3.05 KB) | `BLOCKED` — inspect in Colab Phase A | Deferred |
| Conditional Package C & D | **NOT CREATED** (`0` un-packaged missing features/ckpts) | N/A | Avoid duplicate packaging | Not needed |
"""

    upload_checklist_md = f"""# LUMAE Test-378 Upload & Verification Checklist

1. **Manual Drive Verification (Remote Access Unavailable Locally)**:
   - Open {DRIVE_FOLDER_URL}
   - Confirm `CV_Lumae/artifacts/m1_d200_features/` and `m1_d500_features/` exist (`1,000` NPZs).
   - Confirm `CV_Lumae/artifacts/d1500_benchmark_20261110/packages/lumae_d1000_colab.zip` (`41,623,145` B) and `lumae_d1500_incremental_colab.zip` (`45,201,054` B) exist (upload from `research_outputs/d1000_agent_20261104_01/` and `research_outputs/d1500_agent_20261106_01/` if either is missing).
   - Confirm `CV_Lumae/checkpoints/moment_detr/pretrained/model_best.ckpt` (`16,952,721` B, SHA-256 `{M0}`) exists (upload from `research_outputs/p0_20261103_01/assets/M0_pretrained.ckpt` if missing).
2. **Required New Uploads** into `CV_Lumae/artifacts/d1500_test378_ai_v1/packages/`:
   - [ ] `research_outputs/d1500_test378_ai_v1/packages/lumae_test378_incremental.zip` (`{pkg_a_meta["compressed_bytes"]}` bytes, `{pkg_a_meta["file_count"]}` files, SHA-256 `{pkg_a_sha}`)
   - [ ] `research_outputs/d1500_test378_ai_v1/packages/lumae_colab_benchmark.zip` (`{pkg_b_meta["compressed_bytes"]}` bytes, `{pkg_b_meta["file_count"]}` files, SHA-256 `{pkg_b_sha}`)
3. **Upload Notebook** (also embedded inside `lumae_colab_benchmark.zip`):
   - `research_outputs/d1500_test378_ai_v1/notebooks/LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb` -> `CV_Lumae/notebooks/LUMAE_D1500_Benchmark_3Seeds_Ablation.ipynb`
"""

    validation_md = f"""# LUMAE Test-378 Package Validation Report

- **Release Version**: `{VERSION}`
- **Generated (UTC)**: `{local_inventory["generated_utc"]}`
- **Package C / Package D Status**: Not created (`3,000 / 3,000` base NPZs already covered by Drive D500 + `lumae_d1000_colab.zip` + `lumae_d1500_incremental_colab.zip`; `0` verified local checkpoints absent from Drive inventory).

| Archive | Files | Size (bytes) | SHA-256 | `testzip` | Non-Portable Refs |
| --- | ---: | ---: | --- | --- | ---: |
| `packages/lumae_test378_incremental.zip` | {pkg_a_meta["file_count"]} | {pkg_a_meta["compressed_bytes"]} | `{pkg_a_sha}` | `{pkg_a_check["testzip"]}` | `{pkg_a_check["non_portable_refs"]}` |
| `packages/lumae_colab_benchmark.zip` | {pkg_b_meta["file_count"]} | {pkg_b_meta["compressed_bytes"]} | `{pkg_b_sha}` | `{pkg_b_check["testzip"]}` | `{pkg_b_check["non_portable_refs"]}` |
"""

    drive_manifest = {
        "benchmark_version": VERSION,
        "generated_utc": local_inventory["generated_utc"],
        "drive_upload_root": DRIVE_ARTIFACT_ROOT,
        "remote_verification_status": "UNVERIFIED_NO_DRIVE_AUTH",
        "upload_required": [
            {
                "name": "lumae_test378_incremental.zip",
                "local_relative_path": "packages/lumae_test378_incremental.zip",
                "drive_destination": f"{DRIVE_ARTIFACT_ROOT}/packages/lumae_test378_incremental.zip",
                "sha256": pkg_a_sha,
                "size_bytes": pkg_a_meta["compressed_bytes"],
                "file_count": pkg_a_meta["file_count"],
                "role": "Test-378 JSONLs, provenance, 228 validated new CLIP text NPZs, and freeze manifest.",
            },
            {
                "name": "lumae_colab_benchmark.zip",
                "local_relative_path": "packages/lumae_colab_benchmark.zip",
                "drive_destination": f"{DRIVE_ARTIFACT_ROOT}/packages/lumae_colab_benchmark.zip",
                "sha256": pkg_b_sha,
                "size_bytes": pkg_b_meta["compressed_bytes"],
                "file_count": pkg_b_meta["file_count"],
                "role": "Moment-DETR code, runner, frozen Train/Val/Test splits, scripts, matrix, and Colab notebook.",
            },
        ],
        "conditional_packages": {
            "package_c_created": False,
            "package_c_reason": "No missing D1500 feature gaps beyond existing lumae_d1000_colab.zip, lumae_d1500_incremental_colab.zip, and Drive D500.",
            "package_d_created": False,
            "package_d_reason": "No verified local checkpoints require separate packaging (M0 verified matches provisional Drive copy; QR-DETR/UniVTG are not present locally).",
        },
        "zip_checks": [pkg_a_check, pkg_b_check],
    }

    _write_bytes(out_dir / "DRIVE_UPLOAD_AUDIT.md", drive_upload_audit_md.encode("utf-8"))
    _write_bytes(out_dir / "UPLOAD_CHECKLIST.md", upload_checklist_md.encode("utf-8"))
    _write_bytes(out_dir / "PACKAGE_VALIDATION_REPORT.md", validation_md.encode("utf-8"))
    _write_json(out_dir / "DRIVE_UPLOAD_MANIFEST.json", drive_manifest)

    tracked = sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    sha_lines = [f"{_sha256_file(out_dir / rel)}  {rel}" for rel in tracked]
    _write_bytes(out_dir / "SHA256SUMS.txt", ("\n".join(sha_lines) + "\n").encode("utf-8"))

    # Confirm zero local Windows paths leaked inside any packaged ZIPs or generated out_dir text files.
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".txt", ".ipynb"}:
            text = p.read_text(encoding="utf-8")
            if p.name in {"original_test150.jsonl", "legacy_test30.jsonl", "new_test120.jsonl"}:
                continue
            if p.name == "test378.jsonl":
                text = "\n".join([l for l in text.splitlines() if l.strip()][150:])
            if FORBIDDEN_LOCAL_PATH_RE.search(text):
                raise ValueError(f"Non-portable local path leaked into {p}")

    return {
        "version": VERSION,
        "output_dir": str(out_dir),
        "packages": {
            "lumae_test378_incremental.zip": {**pkg_a_meta, "sha256": pkg_a_sha, **pkg_a_check},
            "lumae_colab_benchmark.zip": {**pkg_b_meta, "sha256": pkg_b_sha, **pkg_b_check},
        },
        "counts": integrity_report["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate LUMAE Test-378 freeze and Colab packages.")
    parser.add_argument("--work-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    result = build_and_package_test378(work_dir=args.work_dir, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
