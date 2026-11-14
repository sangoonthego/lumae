"""Canonical ranked seconds, pinned MR metrics and video-cluster uncertainty."""

from collections import defaultdict
import math
import random
import statistics

from experiments.M1_lumae_ads import cached_runner as runner
from data_process.lumae_scale.models import load_jsonl
from .annotation import tiou
from .parent import SOURCE, valid_window
from .release import feature_manifest, text_feature, validate_release
from .training import recipe


def moment_dataset(parent, work, split, features=None, annotations=None):
    if features is None:
        if split == "test450":
            annotations = work / "release/test450.jsonl"
            features = parent.features(("test",))
            for row in load_jsonl(work / "release/new_test300.jsonl"):
                features["text"][row["qid"]] = text_feature(work, row)
        else:
            annotations = parent.frozen / f"{split}.jsonl"
            features = parent.features(("val" if split == "val" else "test",))
    manifest = feature_manifest(features, lambda f: str(f.path))
    for group in features.values():
        for feature in group.values():
            feature.array()
    return runner.dataset_type(SOURCE / "upstream")(
        annotations, work, manifest, recipe(parent, 42),
        resolve_feature=lambda kind, ident: features[kind][str(ident)].path,
    )


def predict_moment(parent, work, split, checkpoint, device):
    import time
    import torch
    from torch.utils.data import DataLoader

    root = SOURCE / "upstream"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    runner.bind_upstream(root)
    from moment_detr.start_end_dataset import prepare_batch_inputs, start_end_collate
    from moment_detr.span_utils import span_cxw_to_xx

    dataset = moment_dataset(parent, work, split)
    if checkpoint is None:
        checkpoint = SOURCE / "assets/M0_pretrained.ckpt"
        model, _, _ = runner.load_model(root, checkpoint, device)
    else:
        from argparse import Namespace
        from moment_detr.model import build_model

        with torch.serialization.safe_globals([Namespace]):
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        opt = saved["opt"]
        opt.device = torch.device(device)
        model, _ = build_model(opt)
        model.load_state_dict(saved["model"], strict=True)
        model.to(device)
    model.eval()
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = 0.0
    output = []
    loader = DataLoader(dataset, batch_size=4, collate_fn=start_end_collate)
    with torch.inference_mode():
        for meta, data in loader:
            inputs, _ = prepare_batch_inputs(data, torch.device(device))
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = model(**inputs)
            if cuda:
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            windows = span_cxw_to_xx(result["pred_spans"]).cpu().tolist()
            scores = result["pred_logits"].softmax(-1)[..., 0].cpu().tolist()
            for row, spans, confidences in zip(meta, windows, scores):
                length = row["duration"]
                ranked = sorted(zip(spans, confidences), key=lambda pair: -pair[1])
                clipped = []
                for (a, b), score in ranked:
                    interval = [max(0.0, a * length), min(length, b * length)]
                    if valid_window(interval, length):
                        clipped.append([*interval, score])
                output.append({"qid": row["qid"], "vid": row["vid"], "pred_relevant_windows": clipped})
    return output, {
        "device": device, "forward_ms_per_query": elapsed * 1000 / len(output),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
    }


def load_predictions(rows, predictions):
    gt = {str(row["qid"]): row for row in rows}
    canonical = {}
    for pred in predictions:
        qid = str(pred["qid"])
        if qid not in gt or qid in canonical or pred["vid"] != gt[qid]["vid"]:
            raise ValueError(f"Duplicate or incorrect prediction identity: {qid}")
        windows = pred["pred_relevant_windows"]
        for window in windows:
            if (len(window) != 3 or not valid_window(window[:2], gt[qid]["duration"])
                    or type(window[2]) not in (float, int) or not math.isfinite(window[2])):
                raise ValueError(f"Invalid canonical prediction: {qid}")
        canonical[qid] = sorted(windows, key=lambda w: -w[2])
    if set(canonical) != set(gt):
        raise ValueError("Prediction QIDs must exactly match the selected benchmark")
    return canonical


def mean_percent(values):
    return round(100 * statistics.mean(values), 2) if values else None


def metrics(rows, predictions, *, bootstrap=1000, seed=20261114, k=5):
    import numpy as np

    runner.bind_upstream(SOURCE / "upstream")
    from standalone_eval.eval import compute_average_precision_detection_wrapper

    thresholds = np.linspace(0.5, 0.95, 10).round(2)
    grouped = defaultdict(list)
    for row in rows:
        qid = str(row["qid"])
        windows = predictions[qid]
        target = row["relevant_windows"][0]
        overlap = [tiou(window[:2], target) for window in windows]
        top = overlap[0] if overlap else 0.0
        ranked = [{"video-id": qid, "t-start": a, "t-end": b, "score": score} for a, b, score in windows[:10]]
        reference = [{"video-id": qid, "t-start": target[0], "t-end": target[1]}]
        _, ap = compute_average_precision_detection_wrapper((qid, reference, ranked), thresholds)
        grouped[row["vid"]].append((top, max(overlap[:k], default=0), float(ap.mean()), target[1] - target[0] <= 2))

    def summarize(items):
        return {
            "R1@0.5": mean_percent([int(top >= 0.5) for top, _, _, _ in items]),
            "R1@0.7": mean_percent([int(top >= 0.7) for top, _, _, _ in items]),
            "mIoU": mean_percent([top for top, _, _, _ in items]),
            "MR_mAP": mean_percent([ap for _, _, ap, _ in items]),
            f"Oracle@{k}/0.5": mean_percent([int(best >= 0.5) for _, best, _, _ in items]),
            "short_R1@0.5": mean_percent([int(top >= 0.5) for top, _, _, short in items if short]),
        }

    result = summarize([item for group in grouped.values() for item in group])
    if bootstrap:
        rng = random.Random(seed)
        groups = list(grouped.values())
        draws = [summarize([item for group in rng.choices(groups, k=len(groups)) for item in group])
                 for _ in range(bootstrap)]
        intervals = {}
        for metric in result:
            values = [draw[metric] for draw in draws if draw[metric] is not None]
            intervals[metric] = np.percentile(values, [2.5, 97.5]).round(2).tolist() if values else None
        result["video_bootstrap"] = {"seed": seed, "draws": bootstrap, "ci95": intervals}
    return {"query_count": len(rows), "video_count": len(grouped), **result}


def evaluate(parent, store, split, predictions, *, bootstrap, seed, k):
    if bootstrap < 0 or k < 1:
        raise ValueError("Bootstrap count must be nonnegative and K must be positive")
    if split == "val":
        reports = {"val150": parent.rows["val"]}
    else:
        reports = {"legacy_test30": parent.rows["legacy_test30"], "other_test120": parent.rows["new_test120"],
                   "original_test150": parent.rows["test"]}
    if split == "test450":
        validate_release(parent, store)
        new = load_jsonl(store.work / "release/new_test300.jsonl")
        reports |= {
            "new_test300": new, "combined_test450": parent.rows["test"] + new,
        }
    canonical = load_predictions(next(reversed(reports.values())), predictions)
    return {name: metrics(rows, canonical, bootstrap=bootstrap, seed=seed, k=k)
            for name, rows in reports.items()}


def aggregate_seeds(reports):
    from .training import SEEDS

    if set(reports) != set(SEEDS):
        raise ValueError("Provide one evaluation report for each registered training seed")
    first = reports[SEEDS[0]]
    return {
        subset: {
            metric: {"mean": round(statistics.mean(values), 2), "sample_std": round(statistics.stdev(values), 2)}
            for metric, value in first[subset].items()
            if type(value) in (float, int) and metric not in ("query_count", "video_count")
            for values in [[reports[seed][subset][metric] for seed in SEEDS]]
        }
        for subset in first
    }
