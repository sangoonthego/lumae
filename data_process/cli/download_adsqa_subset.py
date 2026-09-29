"""CLI utility to acquire a targeted subset of AdsQA product-ad videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
from typing import Any
import urllib.error
import urllib.request

from ..adsqa.candidate_selection import select_candidates
from ..adsqa.video_validation import probe_video_file


def download_single_video(
    cand: dict[str, str],
    out_dir: Path,
    timeout: int,
    user_agent: str,
) -> dict[str, Any]:
    """Download and probe a single video."""
    vid_id = cand["source_video_id"]
    url = cand["source_url"]
    split = cand.get("source_split", "unassigned")
    category = cand.get("category", "Unknown")
    target_filename = f"{vid_id}.mp4"
    dest_file = out_dir / target_filename

    log_entry: dict[str, Any] = {
        "source_video_id": vid_id,
        "source_split": split,
        "source_url": url,
        "category": category,
        "output_filename": target_filename,
        "download_attempt": 1,
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": None,
        "file_size": 0,
        "download_status": "PENDING",
        "error_reason": None,
        "probe_details": None,
        "model_compatibility": "PENDING",
        "duration_seconds": "",
    }

    # Check if already downloaded and valid
    if dest_file.is_file() and dest_file.stat().st_size > 0:
        probe = probe_video_file(dest_file)
        log_entry["file_size"] = dest_file.stat().st_size
        log_entry["http_status"] = 200
        log_entry["probe_details"] = probe
        log_entry["duration_seconds"] = str(probe["duration"])
        if probe["is_model_compatible"]:
            log_entry["download_status"] = "DOWNLOADED"
            log_entry["model_compatibility"] = "COMPATIBLE"
        else:
            log_entry["download_status"] = "DOWNLOADED"
            log_entry["model_compatibility"] = "MODEL_INCOMPATIBLE"
            log_entry["error_reason"] = probe["incompatibility_reason"]
        return log_entry

    headers = {"User-Agent": user_agent}
    req = urllib.request.Request(url, headers=headers)
    temp_dest = dest_file.with_suffix(f".tmp_{threading.get_ident()}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            log_entry["http_status"] = resp.getcode()
            with temp_dest.open("wb") as out_f:
                while chunk := resp.read(65536):
                    out_f.write(chunk)
        temp_dest.replace(dest_file)
        file_sz = dest_file.stat().st_size
        log_entry["file_size"] = file_sz

        probe = probe_video_file(dest_file)
        log_entry["probe_details"] = probe
        log_entry["duration_seconds"] = str(probe["duration"])

        if probe["is_model_compatible"]:
            log_entry["download_status"] = "DOWNLOADED"
            log_entry["model_compatibility"] = "COMPATIBLE"
        else:
            log_entry["download_status"] = "DOWNLOADED"
            log_entry["model_compatibility"] = "MODEL_INCOMPATIBLE"
            log_entry["error_reason"] = probe["incompatibility_reason"]

    except urllib.error.HTTPError as http_err:
        if temp_dest.is_file():
            temp_dest.unlink()
        log_entry["http_status"] = http_err.code
        if http_err.code in (404, 410):
            log_entry["download_status"] = "UNAVAILABLE"
            log_entry["error_reason"] = f"HTTP {http_err.code} Not Found"
        else:
            log_entry["download_status"] = "FAILED"
            log_entry["error_reason"] = f"HTTP {http_err.code}: {http_err.reason}"

    except Exception as exc:
        if temp_dest.is_file():
            temp_dest.unlink()
        log_entry["download_status"] = "FAILED"
        log_entry["error_reason"] = str(exc)

    return log_entry


def download_adsqa_subset(
    candidates_csv: Path | str = "local_data/reports/adsqa/candidate_review.csv",
    output_dir: Path | str = "local_data/raw/adsqa/videos",
    reports_dir: Path | str = "local_data/reports/adsqa",
    limit: int = 50,
    timeout: int = 30,
    workers: int = 6,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Lumae-AdsQA-Downloader/1.0",
) -> dict[str, Any]:
    """Download and probe candidate product-ad videos with concurrent workers."""
    cand_path = Path(candidates_csv)
    out_dir = Path(output_dir)
    rep_dir = Path(reports_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not cand_path.is_file():
        print(f"Candidate review CSV not found at {cand_path}. Generating from raw manifest...")
        select_candidates("local_data/raw/adsqa", cand_path)

    candidates: list[dict[str, str]] = []
    with cand_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        candidates = list(reader)

    # Prioritize ACCEPT candidates
    accepted = [c for c in candidates if c.get("candidate_status") == "ACCEPT"]
    pending = [c for c in candidates if c.get("candidate_status") == "PENDING"]
    pool = accepted + pending

    print(f"Candidate pool: {len(accepted)} ACCEPT, {len(pending)} PENDING. Target acquired: {limit} (workers={workers}).")

    download_logs: list[dict[str, Any]] = []
    acquired_count = 0
    candidate_updates: dict[str, dict[str, Any]] = {}

    # Check how many already acquired
    for f in out_dir.glob("*.mp4"):
        if f.stat().st_size > 0:
            probe = probe_video_file(f)
            if probe["is_model_compatible"]:
                acquired_count += 1
                candidate_updates[f.stem] = {
                    "download_status": "DOWNLOADED",
                    "duration_seconds": str(probe["duration"]),
                }

    print(f"Initial model-compatible videos already on disk: {acquired_count}/{limit}")

    # Process pool in concurrent batches until limit reached
    candidate_iter = iter(pool)
    batch_size = max(1, workers * 2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while acquired_count < limit:
            batch: list[dict[str, str]] = []
            for _ in range(batch_size):
                try:
                    c = next(candidate_iter)
                    batch.append(c)
                except StopIteration:
                    break

            if not batch:
                print("Exhausted all candidates in pool.")
                break

            future_to_cand = {
                executor.submit(download_single_video, c, out_dir, timeout, user_agent): c
                for c in batch
            }

            for future in concurrent.futures.as_completed(future_to_cand):
                res = future.result()
                download_logs.append(res)
                vid = res["source_video_id"]

                candidate_updates[vid] = {
                    "download_status": res["download_status"],
                    "duration_seconds": res.get("duration_seconds", ""),
                }

                if res["download_status"] == "DOWNLOADED" and res["model_compatibility"] == "COMPATIBLE":
                    acquired_count += 1
                    dur_s = res.get("duration_seconds")
                    print(f"[{acquired_count}/{limit}] {vid} — {dur_s}s ({res['file_size']/(1024*1024):.1f}MB, {res.get('category')})")
                elif res["model_compatibility"] == "MODEL_INCOMPATIBLE":
                    print(f"[MODEL_INCOMPATIBLE] {vid} — {res.get('error_reason')}")
                else:
                    print(f"[{res['download_status']}] {vid} — {res.get('error_reason')}")

                if acquired_count >= limit:
                    break

    # Save detailed download log
    log_file = rep_dir / "download_log.json"
    summary_stats = {
        "target_limit": limit,
        "successfully_acquired": acquired_count,
        "total_attempts": len(download_logs),
        "status_breakdown": {
            "DOWNLOADED": sum(1 for d in download_logs if d["download_status"] == "DOWNLOADED" and d["model_compatibility"] == "COMPATIBLE"),
            "MODEL_INCOMPATIBLE": sum(1 for d in download_logs if d.get("model_compatibility") == "MODEL_INCOMPATIBLE"),
            "UNAVAILABLE": sum(1 for d in download_logs if d["download_status"] == "UNAVAILABLE"),
            "FAILED": sum(1 for d in download_logs if d["download_status"] == "FAILED"),
            "SKIPPED": sum(1 for d in download_logs if d["download_status"] == "SKIPPED"),
        },
        "logs": download_logs,
    }
    log_file.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    print(f"Saved download log to: {log_file} (acquired: {acquired_count}).")

    # Update candidate review CSV
    updated_rows: list[dict[str, str]] = []
    for row in candidates:
        v_id = row["source_video_id"]
        if v_id in candidate_updates:
            upd = candidate_updates[v_id]
            row["download_status"] = upd["download_status"]
            if upd.get("duration_seconds"):
                row["duration_seconds"] = upd["duration_seconds"]
        updated_rows.append(row)

    with cand_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        for r in updated_rows:
            writer.writerow(r)
    print(f"Updated candidate review CSV: {cand_path}")

    return summary_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a targeted pilot subset of AdsQA product-ad videos.")
    parser.add_argument(
        "--candidates",
        default="local_data/reports/adsqa/candidate_review.csv",
        help="Path to candidate review CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="local_data/raw/adsqa/videos",
        help="Destination directory for video files",
    )
    parser.add_argument(
        "--reports-dir",
        default="local_data/reports/adsqa",
        help="Directory to save download report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Target number of successfully acquired and validated videos (default: 50, pilot 30-50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent download threads (default: 6)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Download timeout in seconds per attempt (default: 30)",
    )
    args = parser.parse_args()

    try:
        download_adsqa_subset(
            candidates_csv=args.candidates,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
            limit=args.limit,
            workers=args.workers,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Download execution error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
