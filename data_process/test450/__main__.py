"""Run the D1500 Test-450 workflow with frozen inputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from data_process.lumae_scale.models import atomic_json, digest, load_jsonl, read_json
from .annotation import AnnotationStore, progress
from .evaluation import aggregate_seeds, evaluate, predict_moment
from .parent import M0, OUTPUT, Parent, TrainingParent, jsonl_bytes, safe_id, save_bytes, save_json
from .release import build_text_cache, publish, quality, validate_release
from .resume import checkpoint
from .training import MODELS, SEEDS, matrix, preflight


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("audit")
    for name in ("expand-test", "verify-annotations", "record-generation", "record-review", "retire-attempt"):
        sub = commands.add_parser(name)
        sub.add_argument("--vid")
        sub.add_argument("--attempt", type=int, default=1)
        if name.startswith("record-") or name == "retire-attempt":
            sub.add_argument("decision", type=Path)
        else:
            sub.add_argument("--dry-run", action="store_true")
            if name == "expand-test":
                sub.add_argument("--all", action="store_true", help="Stage overview evidence for all frozen Test VIDs")
    sub = commands.add_parser("boundary-evidence")
    sub.add_argument("--vid", required=True)
    sub.add_argument("--attempt", type=int, default=1)
    sub.add_argument("--role", choices=("generation", "review"), required=True)
    sub.add_argument("--range", type=float, nargs=2, required=True)
    for name in ("progress", "build-text-cache", "quality", "publish"):
        sub = commands.add_parser(name)
        sub.add_argument("--agreement", type=float, default=0.70)
        if name in ("quality", "publish"):
            sub.add_argument("--similarity", type=float, default=0.94)
    commands.add_parser("validate-benchmark")
    sub = commands.add_parser("resume", help="Checkpoint accepted rows and return the next visual-review batch")
    sub.add_argument("--batch-size", type=int, choices=range(1, 11), default=10)
    sub.add_argument("--agreement", type=float, default=0.70)
    sub.add_argument("--encode", action="store_true")
    for name in ("prepare-training-matrix", "training-preflight", "run-training"):
        sub = commands.add_parser(name)
        if name == "prepare-training-matrix":
            sub.add_argument("--dry-run", action="store_true")
        else:
            sub.add_argument("--model", choices=MODELS, default="moment_detr")
            sub.add_argument("--seed", choices=SEEDS, type=int, required=True)
        if name == "run-training":
            sub.add_argument("--execute", action="store_true", required=True)
            sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("evaluate")
    sub.add_argument("--split", choices=("val", "test", "test450"), required=True)
    sub.add_argument("--run", required=True)
    sub.add_argument("--external-model", choices=("qd_detr", "univtg"))
    sub.add_argument("--external-seed", choices=SEEDS, type=int)
    sub.add_argument("--measurements", type=Path, help="JSON timing and checkpoint metadata for an external run")
    sub.add_argument("--lock-test-config", action="store_true")
    inputs = sub.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions", type=Path)
    inputs.add_argument("--moment-detr", action="store_true")
    sub.add_argument("--checkpoint", type=Path)
    sub.add_argument("--device", default="cpu")
    sub.add_argument("--bootstrap", type=int, default=1000)
    sub.add_argument("--bootstrap-seed", type=int, default=20261114)
    sub.add_argument("--oracle-k", type=int, default=5)
    sub = commands.add_parser("aggregate-seeds")
    sub.add_argument("--runs", nargs=3, metavar="EVALUATION_RUN", required=True)
    sub.add_argument("--name", required=True)
    return parser


def packet_result(store, args, role):
    function = store.generation_packet if role == "generation" else store.review_packet
    packet = function(args.vid, args.attempt)
    task = store.case(args.vid, args.attempt) / role / "task.json"
    return {"task_path": str(task), "task_sha256": digest(task), "task": packet}


def stage_generation(store, attempt):
    from concurrent.futures import ThreadPoolExecutor

    def stage(vid):
        try:
            store.generation_packet(vid, attempt)
            return vid, None
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            return vid, str(error)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(stage, sorted(store.parent.test)))
    errors = {vid: error for vid, error in results if error}
    return {"status": "FAIL" if errors else "PASS", "packets_ready": len(results) - len(errors),
            "errors": errors}


def execute(args, parent, store):
    work, action = store.work, args.action
    if action == "audit":
        return parent.audit()
    if action == "resume":
        return checkpoint(parent, store, args.agreement, args.batch_size, args.encode)
    if action in ("expand-test", "verify-annotations"):
        role = "generation" if action == "expand-test" else "review"
        if args.dry_run:
            return {"status": "DRY_RUN", "role": role, "target_videos": len(parent.test),
                    "target_new_queries": 2 * len(parent.test), "vid": args.vid}
        if action == "expand-test" and args.all:
            if args.vid:
                raise ValueError("Choose --all or --vid")
            return stage_generation(store, args.attempt)
        if not args.vid:
            raise ValueError("Supply --vid for each evidence packet")
        return packet_result(store, args, role)
    if action in ("record-generation", "record-review"):
        store.record(args.vid, args.attempt, action.removeprefix("record-"), read_json(args.decision))
        return {"status": "RECORDED", "vid": args.vid, "attempt": args.attempt}
    if action == "retire-attempt":
        store.retire(args.vid, args.attempt, read_json(args.decision))
        return {"status": "RETIRED", "vid": args.vid, "attempt": args.attempt}
    if action == "boundary-evidence":
        return store.boundary_packet(args.vid, args.attempt, args.role, *args.range)
    if action in ("progress", "build-text-cache", "quality"):
        rows, sidecar, states = store.accepted(args.agreement)
        if action == "progress":
            return progress(rows, sidecar, states)
        if action == "quality":
            features, checks = quality(parent, work, rows, args.similarity)
            return checks | {"checked_new_queries": len(rows),
                             "new_text_bytes": sum(features["text"][r["qid"]].path.stat().st_size for r in rows)}
        return {"status": "PASS", "text_features": build_text_cache(work, rows)}
    if action == "publish":
        result = publish(parent, store, args.agreement, args.similarity)
        return {"status": "PUBLISHED", "counts": result["counts"]}
    if action == "validate-benchmark":
        if not (work / "release/manifest.json").is_file():
            rows, _, states = store.accepted()
            raise RuntimeError(f"Release absent; {len(rows)}/300 visually accepted queries and "
                               f"{sum(s['status'] == 'UNRESOLVED' for s in states.values())} unresolved videos")
        return validate_release(parent, store)
    if action == "prepare-training-matrix":
        plan = matrix(parent, work)
        if not args.dry_run:
            atomic_json(work / "training_matrix.json", plan)
        return plan
    if action in ("training-preflight", "run-training"):
        if args.model != "moment_detr":
            entry = next(r for r in matrix(parent, work)["runs"] if r["model"] == args.model and r["seed"] == args.seed)
            raise RuntimeError(" ".join(entry["blockers"]))
        return preflight(parent, work, args.seed, action == "run-training", getattr(args, "resume", False))
    if action == "aggregate-seeds":
        reports = [read_json(work / "evaluations" / safe_id(run) / "report.json") for run in args.runs]
        if len({(r["split"], r["model"]) for r in reports}) != 1 or any(not r["model"] for r in reports):
            raise ValueError("Seed reports must use the same registered model and split")
        result = aggregate_seeds({r["training_seed"]: r["metrics"] for r in reports})
        save_json(work / "evaluations" / f"{safe_id(args.name)}_seed_summary.json", result)
        return result

    if args.split != "val" and not args.lock_test_config:
        raise ValueError("Confirm finalized model, checkpoint and post-processing with --lock-test-config")
    if args.split == "test450":
        validate_release(parent, store)
    result_dir = work / "evaluations" / safe_id(args.run)
    if (result_dir / "report.json").exists():
        raise ValueError("Use a fresh evaluation run name")
    if args.predictions and args.checkpoint:
        raise ValueError("--checkpoint requires --moment-detr")
    if args.moment_detr and (args.external_model or args.external_seed or args.measurements):
        raise ValueError("External run metadata requires --predictions")
    if bool(args.external_model) != bool(args.external_seed):
        raise ValueError("Specify both external model and training seed")
    timing, model, seed = None, None, None
    checkpoint_hash = None
    if args.moment_detr:
        model = "moment_detr"
        for registered_seed in SEEDS:
            planned = work / "runs/moment_detr" / f"seed_{registered_seed}/best.ckpt"
            if args.checkpoint and args.checkpoint.resolve() == planned:
                seed = registered_seed
        predictions, timing = predict_moment(parent, work, args.split, args.checkpoint, args.device)
        checkpoint_hash = digest(args.checkpoint) if args.checkpoint else M0
    else:
        predictions = load_jsonl(args.predictions)
        model, seed = args.external_model, args.external_seed
        if args.measurements:
            timing = read_json(args.measurements)
            checkpoint_hash = timing.get("checkpoint_sha256")
    report = {
        "split": args.split, "model": model, "training_seed": seed, "timing": timing,
        "metrics": evaluate(parent, store, args.split, predictions, bootstrap=args.bootstrap,
                            seed=args.bootstrap_seed, k=args.oracle_k),
        "checkpoint_sha256": checkpoint_hash,
        "test_configuration_locked": args.split != "val",
    }
    save_bytes(result_dir / "predictions.jsonl", jsonl_bytes(predictions))
    save_json(result_dir / "report.json", report)
    return report


def main():
    parser = cli()
    args = parser.parse_args()
    work = OUTPUT.resolve()
    sys.dont_write_bytecode = True
    try:
        training_actions = {"prepare-training-matrix", "training-preflight", "run-training", "aggregate-seeds"}
        train_only = args.action in training_actions or (args.action == "evaluate" and args.split == "val")
        parent = TrainingParent() if train_only else Parent()
        store = AnnotationStore(parent, work)
        writers = {"expand-test", "verify-annotations", "boundary-evidence", "record-generation",
                   "record-review", "retire-attempt", "build-text-cache", "publish", "resume"}
        if args.action in writers and not getattr(args, "dry_run", False):
            from data_process.lumae_scale.acquisition import run_lock

            with run_lock(work / "annotation.lock"):
                result = execute(args, parent, store)
        else:
            result = execute(args, parent, store)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "BLOCKED" if isinstance(error, RuntimeError) else "FAIL",
                          "reason": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if isinstance(result, dict) and result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
