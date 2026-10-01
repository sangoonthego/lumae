"""Colab runner for the frozen LUMAE Ads v1 Moment-DETR experiment.

Uses upstream CLIP features and model weights. Only temporal spans are targets;
saliency labels are never synthesized. Validation labels are AI pseudo labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import zipfile


VERSION = "lumae_ads_v1"
HANDOFF = Path("local_data/manifests/lumae_ads_v1_training_handoff.json")
CONFIG = Path("experiments/M1_lumae_ads/config.json")
VIDEOS = Path("local_data/raw/adsqa/videos")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(root: Path, *, check_videos: bool = True) -> tuple[dict, dict, dict]:
    """Verify the immutable training handoff and return rows by split."""
    root = root.resolve()
    handoff = _json(root / HANDOFF)
    if handoff.get("dataset_version") != VERSION or handoff.get("has_saliency_gt") is not False:
        raise ValueError("Wrong dataset version or saliency declaration")
    if handoff.get("label_provenance_counts") != {"human_verified": 18, "ai_pseudo_labeled": 30}:
        raise ValueError("Training handoff provenance counts differ")
    manifest_path = root / handoff["dataset_manifest_path"]
    if sha256(manifest_path) != handoff["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest checksum mismatch")
    manifest = _json(manifest_path)
    if (manifest.get("total_samples"), manifest.get("human_verified_samples"),
            manifest.get("ai_pseudo_labeled_samples")) != (48, 18, 30):
        raise ValueError("Frozen dataset provenance counts differ")
    dataset_dir = manifest_path.parent
    checks = (dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    if len(checks) != 7:
        raise ValueError("Frozen dataset checksum coverage is incomplete")
    for line in checks:
        digest, name = line.split("  ", 1)
        if sha256(dataset_dir / name) != digest:
            raise ValueError(f"Frozen dataset file changed: {name}")
    if sha256(dataset_dir / f"{VERSION}_all.jsonl") != manifest["dataset_sha256"]:
        raise ValueError("Frozen all-sample hash mismatch")
    for name, digest in manifest["source_artifact_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Source annotation changed: {name}")
    audit = _json(dataset_dir / "annotation_audit.json")
    if len(audit.get("pseudo_label_artifacts", {})) != 30:
        raise ValueError("Pseudo-label audit coverage is incomplete")
    for item in audit["pseudo_label_artifacts"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Pseudo-label artifact changed: {item['path']}")
    rows = {}
    for split in SPLITS:
        spec = handoff[f"{split}_manifest"]
        path = root / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"{split} JSONL checksum mismatch")
        rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [len(rows[name]) for name in SPLITS] != [34, 7, 7]:
        raise ValueError("Unexpected frozen split counts")
    all_rows = [row for split in SPLITS for row in rows[split]]
    if len({row["qid"] for row in all_rows}) != 48:
        raise ValueError("Duplicate or missing sample ID")
    vids = {split: {row["vid"] for row in rows[split]} for split in SPLITS}
    if any(vids[a] & vids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Video leakage between frozen splits")
    split_manifest = _json(manifest_path.parent / "split_manifest.json")
    forced = set(split_manifest["forced_train_ids"])
    if not forced <= {row["qid"] for row in rows["train"]}:
        raise ValueError("Forced-train sample outside train")
    expected = {
        "train": Counter({"HUMAN_VERIFIED": 18, "AI_PSEUDO_LABELED": 16}),
        "val": Counter({"AI_PSEUDO_LABELED": 7}),
        "test": Counter({"AI_PSEUDO_LABELED": 7}),
    }
    for split in SPLITS:
        if Counter(row["metadata"]["review_provenance"] for row in rows[split]) != expected[split]:
            raise ValueError(f"Unexpected {split} label provenance")
    for row in all_rows:
        window = row["relevant_windows"]
        if (not row["query"].strip() or len(window) != 1 or len(window[0]) != 2
                or not all(math.isfinite(v) for v in (*window[0], row["duration"]))
                or not 0 <= window[0][0] < window[0][1] <= row["duration"]
                or row["saliency_scores"] is not None or row["relevant_clip_ids"] is not None):
            raise ValueError(f"Invalid temporal record: {row['qid']}")
        if check_videos:
            video = root / VIDEOS / row["metadata"]["video_filename"]
            source = manifest["source_video_sha256"][row["qid"]]
            if not video.is_file() or sha256(video) != source["sha256"]:
                raise ValueError(f"Missing or changed source video: {row['qid']}")
    return handoff, manifest, rows


def _upstream(moment_root: Path):
    if not (moment_root / "moment_detr" / "model.py").is_file():
        raise FileNotFoundError("Official Moment-DETR checkout is missing")
    sys.path.insert(0, str(moment_root.resolve()))
    from run_on_video.data_utils import ClipFeatureExtractor
    from moment_detr.model import build_model
    from moment_detr.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
    return ClipFeatureExtractor, build_model, StartEndDataset, start_end_collate, prepare_batch_inputs


def package_colab(root: Path, destination: Path) -> dict:
    """Create a portable, verified input bundle without changing frozen files."""
    root = root.resolve()
    handoff, manifest, _ = preflight(root)
    dataset_dir = (root / handoff["dataset_manifest_path"]).parent
    paths = [root / HANDOFF, root / "03_Lumae_M1_ProductAds.ipynb",
             root / "experiments/M1_lumae_ads/train_frozen.py",
             root / "experiments/M1_lumae_ads/config.json",
             root / "experiments/M1_lumae_ads/README.md"]
    paths.extend(sorted(dataset_dir.iterdir()))
    paths.extend(root / name for name in manifest["source_artifact_sha256"])
    paths.extend(root / VIDEOS / item["video_filename"]
                 for item in manifest["source_video_sha256"].values())
    audit = _json(dataset_dir / "annotation_audit.json")
    for item in audit["pseudo_label_artifacts"].values():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Pseudo-label audit hash mismatch: {item['path']}")
        paths.append(path)
    if len(paths) != len({path.resolve() for path in paths}) or any(not path.is_file() for path in paths):
        raise ValueError("Portable bundle inputs are missing or duplicated")
    destination = destination.resolve()
    if destination.is_relative_to(dataset_dir) or destination.is_relative_to(root / VIDEOS):
        raise ValueError("Bundle destination must not be inside frozen inputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, arcname=path.relative_to(root).as_posix())
    return {"path": str(destination), "sha256": sha256(destination),
            "file_count": len(paths), "bytes": destination.stat().st_size,
            "dataset_sha256": manifest["dataset_sha256"]}


def extract_features(root: Path, moment_root: Path, feature_root: Path) -> dict:
    """Extract official 2-second CLIP image/text features for train and val only."""
    import numpy as np
    import torch

    handoff, manifest, rows = preflight(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a CUDA Colab runtime")
    extractor_type, _, _, _, _ = _upstream(moment_root)
    feature_root = feature_root.resolve()
    video_dir, text_dir = feature_root / "clip_features", feature_root / "clip_text_features"
    video_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    existing = feature_root / "feature_manifest.json"
    if existing.exists():
        raise ValueError("Feature manifest already exists; use a new feature directory for another extraction")
    extractor = extractor_type(framerate=0.5, size=224, centercrop=True,
                               model_name_or_path="ViT-B/32", device="cuda")
    features = {"dataset_sha256": manifest["dataset_sha256"],
                "dataset_manifest_sha256": handoff["dataset_manifest_sha256"],
                "extractor": "official Moment-DETR ClipFeatureExtractor ViT-B/32 0.5 fps",
                "video": {}, "text": {}}
    for row in rows["train"] + rows["val"]:
        vid, qid = row["vid"], row["qid"]
        video_path = root / VIDEOS / row["metadata"]["video_filename"]
        video_feature = extractor.encode_video(str(video_path), bsz=16).detach().float().cpu().numpy()
        if video_feature.ndim != 2 or video_feature.shape[1] != 512 or not 1 <= len(video_feature) <= 75:
            raise ValueError(f"Invalid CLIP video feature shape: {qid}, {video_feature.shape}")
        video_out = video_dir / f"{vid}.npz"
        np.savez_compressed(video_out, features=video_feature)
        text_feature = extractor.encode_text([row["query"]], bsz=1)[0].detach().float().cpu().numpy()
        if text_feature.ndim != 2 or text_feature.shape[1] != 512 or len(text_feature) < 1:
            raise ValueError(f"Invalid CLIP text feature shape: {qid}, {text_feature.shape}")
        text_out = text_dir / f"qid{qid}.npz"
        np.savez_compressed(text_out, last_hidden_state=text_feature[:32])
        features["video"][vid] = {"path": str(video_out), "sha256": sha256(video_out),
                                   "shape": list(video_feature.shape)}
        features["text"][qid] = {"path": str(text_out), "sha256": sha256(text_out),
                                  "shape": list(text_feature[:32].shape)}
    existing.write_text(json.dumps(features, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return features


def _verify_features(feature_root: Path, manifest: dict, rows: dict) -> tuple[Path, Path]:
    features = _json(feature_root / "feature_manifest.json")
    if features.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("Features belong to another dataset freeze")
    for row in rows["train"] + rows["val"]:
        for key, name in (("video", row["vid"]), ("text", row["qid"])):
            item = features[key][name]
            path = Path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"Missing or changed {key} feature: {name}")
    return feature_root / "clip_features", feature_root / "clip_text_features"


def train(root: Path, moment_root: Path, checkpoint_path: Path, feature_root: Path,
          results_root: Path, *, epochs: int = 50, batch_size: int = 4,
          learning_rate: float = 1e-5, patience: int = 10, seed: int = 20260930) -> dict:
    """Fine-tune only on train; choose best weights on pseudo-labeled val."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    root, feature_root = root.resolve(), feature_root.resolve()
    handoff, manifest, rows = preflight(root)
    experiment_config = _json(root / CONFIG)
    if experiment_config.get("dataset_version") != VERSION:
        raise ValueError("M1 config targets another dataset version")
    video_dir, text_dir = _verify_features(feature_root, manifest, rows)
    if not torch.cuda.is_available():
        raise RuntimeError("M1 fine-tuning requires a CUDA Colab runtime")
    if not checkpoint_path.is_file() or epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Trusted parent checkpoint and positive training settings are required")
    checkpoint_sha = sha256(checkpoint_path)
    # PyTorch checkpoints contain an argparse Namespace; load only a checkpoint
    # supplied from a trusted local source, never from arbitrary user uploads.
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opt = parent["opt"]
    if (opt.ctx_mode != "video_tef" or opt.v_feat_dim != 514 or opt.t_feat_dim != 512
            or opt.max_v_l != 75 or opt.clip_length != 2 or opt.span_loss_type != "l1"):
        raise ValueError("Parent checkpoint is not compatible with CLIP-only 2-second M1")
    _, build_model, StartEndDataset, start_end_collate, prepare_batch_inputs = _upstream(moment_root)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt.device = torch.device("cuda")
    opt.max_q_l = handoff["max_query_length"]
    opt.lw_saliency = handoff["recommended_saliency_loss_weight"]
    opt.span_loss_coef = experiment_config["training"]["loss_weights"]["span_loss"]
    opt.giou_loss_coef = experiment_config["training"]["loss_weights"]["giou_loss"]
    model, criterion = build_model(opt)
    model.load_state_dict(parent["model"], strict=True)
    model.to(opt.device)
    criterion.to(opt.device)
    if criterion.weight_dict.get("loss_saliency") != 0.0:
        raise ValueError("Saliency loss was not disabled")

    class SpanOnlyDataset(Dataset):
        def __init__(self, path: Path):
            self.base = StartEndDataset(
                dset_name="hl", data_path=str(path), v_feat_dirs=[str(video_dir)],
                q_feat_dir=str(text_dir), q_feat_type="last_hidden_state",
                max_q_l=32, max_v_l=75, ctx_mode="video_tef", clip_len=2,
                span_loss_type="l1", load_labels=False,
            )

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            item = self.base[index]
            meta, model_inputs = item["meta"], item["model_inputs"]
            model_inputs["span_labels"] = self.base.get_span_labels(
                meta["relevant_windows"], len(model_inputs["video_feat"]))
            return item

    train_data = SpanOnlyDataset(root / handoff["train_manifest"]["path"])
    val_data = SpanOnlyDataset(root / handoff["val_manifest"]["path"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=start_end_collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=start_end_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    run_dir = results_root.resolve() / f"m1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    best = (-1.0, -1.0)
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            if set(targets) != {"span_labels"}:
                raise ValueError("Unexpected non-span training target")
            outputs = model(**inputs)
            loss_map = criterion(outputs, targets)
            loss = sum(value * criterion.weight_dict[key] for key, value in loss_map.items()
                       if key in criterion.weight_dict)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, _ = prepare_batch_inputs(batch[1], opt.device)
                outputs = model(**inputs)
                chosen = outputs["pred_logits"].softmax(-1)[..., 0].argmax(-1)
                for index, meta in enumerate(batch[0]):
                    center, width = outputs["pred_spans"][index, chosen[index]].tolist()
                    duration = meta["duration"]
                    start = max(0.0, min(duration, (center - width / 2) * duration))
                    end = max(0.0, min(duration, (center + width / 2) * duration))
                    gt_start, gt_end = meta["relevant_windows"][0]
                    intersection = max(0.0, min(end, gt_end) - max(start, gt_start))
                    union = max(end, gt_end) - min(start, gt_start)
                    ious.append(intersection / union if union > 0 else 0.0)
        score = (sum(iou >= 0.5 for iou in ious) / len(ious), sum(ious) / len(ious))
        entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses),
                 "val_pseudo_r1_at_0_5": score[0], "val_pseudo_mean_iou": score[1]}
        history.append(entry)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "opt": opt, "epoch": epoch - 1,
                        "parent_checkpoint_sha256": checkpoint_sha,
                        "dataset_sha256": manifest["dataset_sha256"]}, run_dir / "model_best.ckpt")
        else:
            stale += 1
            if stale >= patience:
                break
    result = {
        "status": "TRAINED_PENDING_FINAL_EVALUATION", "dataset_version": VERSION,
        "dataset_sha256": manifest["dataset_sha256"],
        "parent_checkpoint_sha256": checkpoint_sha,
        "best_checkpoint": str(run_dir / "model_best.ckpt"),
        "best_checkpoint_sha256": sha256(run_dir / "model_best.ckpt"),
        "best_epoch": best_epoch, "best_val_pseudo_r1_at_0_5": best[0],
        "best_val_pseudo_mean_iou": best[1],
        "validation_label_provenance": f"{len(rows['val'])} AI_PSEUDO_LABELED samples; no human verified val labels",
        "pseudo_label_quality_warning": handoff.get("annotation_warning",
            "All 30 v1 pseudo intervals have LOW semantic confidence and zero ranking margin."),
        "test_split_used": False,
        "training_label_provenance": dict(Counter(r["metadata"]["review_provenance"].lower()
                                                   for r in rows["train"])),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "package", "features", "train"))
    parser.add_argument("--lumae-root", type=Path, required=True)
    parser.add_argument("--moment-detr-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.action == "preflight":
        handoff, manifest, rows = preflight(args.lumae_root)
        print(json.dumps({"dataset_version": handoff["dataset_version"],
                          "dataset_sha256": manifest["dataset_sha256"],
                          "split_counts": {name: len(rows[name]) for name in SPLITS},
                          "validation_label_provenance": "AI_PSEUDO_LABELED_ONLY"}, indent=2))
    elif args.action == "package":
        if args.bundle_path is None:
            parser.error("package requires --bundle-path")
        print(json.dumps(package_colab(args.lumae_root, args.bundle_path), indent=2))
    elif args.action == "features":
        if args.moment_detr_root is None or args.feature_root is None:
            parser.error("features requires --moment-detr-root and --feature-root")
        features = extract_features(args.lumae_root, args.moment_detr_root, args.feature_root)
        print(json.dumps({"video_features": len(features["video"]),
                          "text_features": len(features["text"])}, indent=2))
    else:
        if any(value is None for value in (
            args.moment_detr_root, args.checkpoint, args.feature_root, args.results_root)):
            parser.error("train requires --moment-detr-root, --checkpoint, --feature-root, --results-root")
        print(json.dumps(train(args.lumae_root, args.moment_detr_root, args.checkpoint,
                               args.feature_root, args.results_root, epochs=args.epochs,
                               batch_size=args.batch_size, learning_rate=args.learning_rate), indent=2))


if __name__ == "__main__":
    main()
