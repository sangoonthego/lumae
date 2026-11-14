"""Frozen CLIP text-cache delta, diversity checks and versioned publication."""

from collections import Counter
from io import BytesIO
from itertools import combinations
import sys

from data_process.lumae_scale.models import ROOT, digest, load_jsonl, read_json
from experiments.M1_lumae_ads import cached_runner
from .annotation import normalized, progress
from .parent import (
    CANONICAL, Feature, KEYS, VERSION, duration_distribution, jsonl_bytes, require_hash,
    save_bytes, save_json, sha_bytes, storage_path,
)

EXTRACTOR = storage_path("E:/LUMAE/work/d1500/d1500_agent_20261106_01/feature_extractor_source")
WEIGHTS = ROOT / "local_data/cache/clip_weights/ViT-B-32.pt"
WEIGHTS_HASH = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
EXTRACTOR_HASHES = {
    "run_on_video/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "run_on_video/clip/__init__.py": "f6ddc7cbe2e119c937398de04e4d064976cc0110c5618bc7312fa595967ae3b4",
    "run_on_video/clip/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "run_on_video/clip/clip.py": "479207dfc7562501ff231f98ca90b0e61569e5f7d5b985a0046e0d746ccc542a",
    "run_on_video/clip/model.py": "e8e189318c30073ee3ed96cff5a606691738d10d529a7bef1b943f62f0030c41",
    "run_on_video/clip/simple_tokenizer.py": "d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad",
    "run_on_video/data_utils.py": "c94810c048888af95496b56878adf93e600fb631d5aa7d319d7e5ce3404b7fb6",
}


def text_path(work, row):
    return work / "text_features" / row["qid"] / f"{sha_bytes(row['query'].encode())}.npz"


def text_feature(work, row):
    file = text_path(work, row)
    binding = read_json(file.with_suffix(".json"))
    if binding["query_sha256"] != sha_bytes(row["query"].encode()) or binding["weights_sha256"] != WEIGHTS_HASH:
        raise ValueError(f"Stale text feature binding: {file}")
    return Feature("text", row["qid"], file, binding["sha256"])


def build_text_cache(work, rows, extractor_root=EXTRACTOR, weights=WEIGHTS):
    if not rows:
        return 0
    import numpy as np
    import torch

    root = extractor_root.resolve()
    for name, expected in EXTRACTOR_HASHES.items():
        require_hash(root / name, expected)
    require_hash(weights, WEIGHTS_HASH)
    if any(name == "run_on_video" or name.startswith("run_on_video.") for name in sys.modules):
        raise ValueError("Load the pinned extractor in a fresh process")
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    from run_on_video.clip.clip import _tokenizer, load, tokenize

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    pending = []
    for row in rows:
        content_tokens = len(_tokenizer.encode(row["query"]))
        if content_tokens > 30:
            raise ValueError(f"Query {row['qid']} has {content_tokens} content tokens. "
                             "Revise and review it before CLIP would truncate it.")
        file = text_path(work, row)
        if file.with_suffix(".json").is_file():
            text_feature(work, row).array()
        else:
            pending.append((row, file))
    if pending:
        model, _ = load(str(weights.resolve()), device="cpu", jit=False)
        model.eval()
        # Stable batch shape also makes interruption between NPZ and binding safe.
        for row, file in pending:
            tokens = tokenize([row["query"]], context_length=77)
            with torch.inference_mode():
                output = model.encode_text(tokens)["last_hidden_state"][0].float().numpy()
            length = int((tokens != 0).sum())
            blob = BytesIO()
            np.savez_compressed(blob, **{KEYS["text"]: output[:length]})
            data = blob.getvalue()
            save_bytes(file, data)
            save_json(file.with_suffix(".json"), {
                "query_sha256": sha_bytes(row["query"].encode()), "weights_sha256": WEIGHTS_HASH,
                "sha256": sha_bytes(data), "token_count": length,
            })
            text_feature(work, row).array()
    return len(rows)


def feature_manifest(features, paths):
    return {
        "canonical_sha256": CANONICAL, "feature_contract": {"tef_in_cache": False},
        **{kind: {ident: paths(feature) for ident, feature in group.items()}
           for kind, group in features.items()},
    }


def diversity(parent, rows, features, threshold=0.94):
    import numpy as np

    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1]")
    originals = {normalized(r["query"]) for split in ("train", "val", "test") for r in parent.rows[split]}
    exact = [{"qid": r["qid"], "against": "original corpus"} for r in rows if normalized(r["query"]) in originals]
    compared = rows + [parent.test[vid] for vid in sorted({r["vid"] for r in rows})]
    vectors = {}
    for row in compared:
        array = features["text"][str(row["qid"])].array()
        value = array[1:-1].mean(axis=0) if len(array) > 2 else array[-1]
        vectors[str(row["qid"])] = value / max(np.linalg.norm(value), 1e-12)
    new_ids = {r["qid"] for r in rows}
    flags = []
    for a, b in combinations(compared, 2):
        qids = [str(a["qid"]), str(b["qid"])]
        if qids[0] not in new_ids or (qids[1] not in new_ids and a["vid"] != b["vid"]):
            continue
        if normalized(a["query"]) == normalized(b["query"]):
            exact.append({"qid": qids[0], "against": qids[1]})
        similarity = float(vectors[qids[0]] @ vectors[qids[1]])
        lexical = set(normalized(a["query"]).split())
        other = set(normalized(b["query"]).split())
        jaccard = len(lexical & other) / len(lexical | other)
        if similarity >= threshold or jaccard >= 0.75:
            pair = "|".join(sorted(qids))
            flags.append({"pair": pair, "cosine": similarity, "word_jaccard": jaccard,
                          "same_video": a["vid"] == b["vid"]})
    return {"exact_duplicates": exact, "flags": flags, "cosine_threshold": threshold}


def quality(parent, work, rows, similarity):
    originals = parent.features(("train", "val", "test"))
    additions = {row["qid"]: text_feature(work, row) for row in rows}
    features = {"video": originals["video"], "text": originals["text"] | additions}
    return features, diversity(parent, rows, features, similarity)


def bundle(parent, store, threshold=0.70, similarity=0.94):
    rows, sidecar, states = store.accepted(threshold)
    report = progress(rows, sidecar, states)
    if Counter(row["vid"] for row in rows) != {vid: 2 for vid in parent.test}:
        raise ValueError(f"Expansion incomplete: {len(rows)} of 300 accepted; {report['unresolved_videos']} unresolved videos")
    features, checks = quality(parent, store.work, rows, similarity)
    for group in features.values():
        for feature in group.values():
            feature.array()
    if checks["exact_duplicates"]:
        raise ValueError("Exact duplicate queries must be revised before release")
    resolutions_path = store.work / "diversity_resolutions.json"
    resolutions = read_json(resolutions_path) if resolutions_path.is_file() else {}
    for flag in checks["flags"]:
        decision = resolutions.get(flag["pair"], {})
        if not all(isinstance(decision.get(k), str) and decision[k].strip() for k in ("reviewer", "reason")):
            raise ValueError(f"Review similarity flag before release: {flag['pair']}")
        if decision.get("distinct_events") is not True:
            raise ValueError(f"Retire the duplicate attempt and inspect a new event: {flag['pair']}")
        flag["resolution"] = decision
    frozen = parent.frozen / "test.jsonl"
    original_bytes = frozen.read_bytes()
    if not original_bytes.endswith(b"\n"):
        raise ValueError("Frozen Test JSONL must end at a complete line boundary")
    checks["duration_buckets"] = {s: duration_distribution(parent.rows[s]) for s in ("train", "val", "test")}
    checks["duration_buckets"]["new_test300"] = duration_distribution(rows)
    checks["duration_buckets"]["combined_test450"] = duration_distribution(parent.rows["test"] + rows)
    report["diversity"] = checks
    report["additional_text_bytes"] = sum(features["text"][row["qid"]].path.stat().st_size for row in rows)
    report["feature_counts"] = {"reused_video": 1500, "reused_text": 1500, "new_text": 300}
    return original_bytes + jsonl_bytes(rows), rows, sidecar, features, report


def publish(parent, store, threshold=0.70, similarity=0.94):
    test, rows, sidecar, features, report = bundle(parent, store, threshold, similarity)
    release = store.work / "release"
    files = {
        "test450.jsonl": test, "new_test300.jsonl": jsonl_bytes(rows),
        "provenance.jsonl": jsonl_bytes(sidecar),
    }
    for name, data in files.items():
        save_bytes(release / name, data)
    manifest = {
        "dataset_version": VERSION, "parent_dataset_version": "lumae_ads_d1500",
        "parent_canonical_sha256": CANONICAL, "parent_split_sha256": parent.hashes,
        "counts": {"videos": 1500, "queries": 1800, "temporal_annotations": 1800, "test_queries": 450,
                   "video_features": 1500, "text_features": 1800},
        "agreement_threshold": threshold, "similarity_threshold": similarity, "quality": report,
        "text_encoder": {"weights_sha256": WEIGHTS_HASH, "commit": cached_runner.UPSTREAM,
                         "context_length": 77, "max_valid_tokens": 32, "extractor_sha256": EXTRACTOR_HASHES},
        "files": {name: sha_bytes(data) for name, data in files.items()},
        "features": {kind: {ident: {"path": str(feature.path.resolve()), "sha256": feature.sha256}
                            for ident, feature in group.items()} for kind, group in features.items()},
    }
    # The manifest is the commit point for the release.
    save_json(release / "manifest.json", manifest)
    return manifest


def validate_release(parent, store):
    release = store.work / "release"
    manifest = read_json(release / "manifest.json")
    if manifest["dataset_version"] != VERSION:
        raise ValueError("Wrong dataset version")
    if (manifest["parent_canonical_sha256"] != CANONICAL or
            manifest["counts"] != {"videos": 1500, "queries": 1800, "temporal_annotations": 1800,
                                   "test_queries": 450, "video_features": 1500, "text_features": 1800}):
        raise ValueError("Unexpected parent identity or release inventory")
    test, rows, sidecar, features, report = bundle(
        parent, store, manifest["agreement_threshold"], manifest["similarity_threshold"],
    )
    expected = {"test450.jsonl": test, "new_test300.jsonl": jsonl_bytes(rows), "provenance.jsonl": jsonl_bytes(sidecar)}
    for name, data in expected.items():
        require_hash(release / name, sha_bytes(data))
        if manifest["files"][name] != sha_bytes(data):
            raise ValueError(f"Release checksum mismatch: {name}")
    for kind, group in features.items():
        actual = {ident: {"path": str(f.path.resolve()), "sha256": f.sha256} for ident, f in group.items()}
        if actual != manifest["features"][kind]:
            raise ValueError(f"Release {kind} bindings differ")
    if report != manifest["quality"] or manifest["parent_split_sha256"] != parent.hashes:
        raise ValueError("Release quality or parent seal differs")
    expanded = load_jsonl(release / "test450.jsonl")
    if len({str(row["qid"]) for row in expanded}) != 450:
        raise ValueError("Expected 450 distinct test QIDs")
    return {"status": "PASS", "test_videos": len(parent.test), "test_queries": len(expanded),
            "new_queries": len(rows), **report["feature_counts"]}
