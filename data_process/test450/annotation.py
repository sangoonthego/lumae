"""Evidence packets and append-only blind annotation intake."""

from collections import Counter
from datetime import datetime
from io import BytesIO
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

from data_process.lumae_scale.models import digest, read_json
from .parent import VERSION, require_hash, safe_id, save_bytes, save_json, valid_window

SOURCE = "AI_PSEUDO_VERIFIED_BY_AI"


def normalized(query):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", query).casefold()))


def tiou(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def frame_sheet(video, first, step, destination, last=None):
    from PIL import Image, ImageDraw

    with tempfile.TemporaryDirectory() as scratch:
        pattern = str(Path(scratch) / "%05d.jpg")
        selection = f"gte(t,{first}+selected_n*{step})"
        if last is not None:
            selection += f"*lte(t,{last})"
        result = subprocess.run(
            ["ffmpeg", "-v", "info", "-nostdin", "-i", str(video), "-map", "0:v:0", "-an",
             "-vf", f"select='{selection}',showinfo,scale=240:135:force_original_aspect_ratio=decrease",
             "-fps_mode", "passthrough", "-threads", "1", pattern],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        paths = sorted(Path(scratch).glob("*.jpg"))
        timestamps = [float(value) for value in re.findall(r"pts_time:([-+0-9.eE]+)", result.stderr)]
        if not paths or len(paths) != len(timestamps):
            raise ValueError(f"Decoded frame/timestamp count mismatch: {video}")
        for offset in range(0, len(paths), 40):
            page = paths[offset:offset + 40]
            canvas = Image.new("RGB", (980, math.ceil(len(page) / 4) * 159), "#151515")
            draw = ImageDraw.Draw(canvas)
            for index, path in enumerate(page):
                x, y = (index % 4) * 245 + 2, (index // 4) * 159
                with Image.open(path) as frame:
                    canvas.paste(frame, (x, y))
                draw.text((x, y + 138), f"{timestamps[offset + index]:.4f} s", fill="white")
            blob = BytesIO()
            canvas.save(blob, format="JPEG", quality=88)
            save_bytes(destination / f"sheet_{offset // 40:03}.jpg", blob.getvalue())
        return timestamps


class AnnotationStore:
    def __init__(self, parent, work):
        self.parent, self.work = parent, work

    def case(self, vid, attempt):
        if vid not in self.parent.test or not 1 <= attempt <= 999:
            raise ValueError("Expected a frozen Test VID and attempt between 1 and 999")
        return self.work / "cases" / safe_id(vid) / f"{attempt:03}"

    def require_open(self):
        if (self.work / "release/manifest.json").is_file():
            raise ValueError("The annotation ledger is sealed for this release")

    def evidence(self, vid, attempt, role):
        self.require_open()
        if role not in ("generation", "review"):
            raise ValueError("Unknown evidence role")
        case = self.case(vid, attempt)
        folder = case / role
        path = folder / "evidence.json"
        if path.exists():
            packet = read_json(path)
            for name, expected in packet["sheets"].items():
                require_hash(folder / name, expected)
            return packet
        row = self.parent.test[vid]
        duration = row["duration"]
        # The review grid is chosen from duration and role alone.
        step = min(0.5 if role == "generation" else 0.4, duration / 3)
        first = 0 if role == "generation" else step / 2
        timestamps = frame_sheet(self.parent.media(vid), first, step, folder)
        packet = {
            "vid": vid, "duration": duration, "role": role, "sampling_seconds": step,
            "video_sha256": row["metadata"]["source_video_sha256"], "timestamps": timestamps,
            "sheets": {p.name: digest(p) for p in sorted(folder.glob("sheet_*.jpg"))},
        }
        save_json(path, packet)
        return packet

    def generation_packet(self, vid, attempt):
        row = self.parent.test[vid]
        case = self.case(vid, attempt)
        packet = self.evidence(vid, attempt, "generation") | {
            "original_query": row["query"], "original_window": row["relevant_windows"][0],
        }
        save_json(case / "generation/task.json", packet)
        return packet

    def review_packet(self, vid, attempt):
        case = self.case(vid, attempt)
        generated = read_json(case / "generation/decision.json")
        packet = self.evidence(vid, attempt, "review") | {
            "queries": [x["query"] for x in generated["events"]],
        }
        save_json(case / "review/task.json", packet)
        return packet

    def boundary_packet(self, vid, attempt, role, start, end):
        self.require_open()
        case = self.case(vid, attempt)
        if role not in ("generation", "review") or not valid_window([start, end], self.parent.test[vid]["duration"]):
            raise ValueError("Boundary evidence requires a role and a valid time range")
        # The blind reviewer chooses review ranges from independent observations.
        task_hash = digest(case / role / "task.json")
        key = f"{start:.6f}_{end:.6f}"
        folder = case / role / "boundary" / key
        timestamps = frame_sheet(self.parent.media(vid), start, 0.1, folder, end)
        packet = {"task_sha256": task_hash, "timestamps": timestamps, "sampling_seconds": 0.1,
                  "sheets": {p.name: digest(p) for p in sorted(folder.glob("sheet_*.jpg"))}}
        save_json(folder / "evidence.json", packet)
        return packet | {"path": str(folder / "evidence.json"), "key": key}

    @staticmethod
    def validate_decision(decision, packet):
        if decision.get("task_sha256") != packet["task_sha256"]:
            raise ValueError("Decision must identify the exact inspected task")
        if not all(isinstance(decision.get(k), str) and decision[k].strip()
                   for k in ("agent", "model", "session", "timestamp_utc")):
            raise ValueError("Missing agent, model, session or UTC timestamp")
        recorded = datetime.fromisoformat(decision["timestamp_utc"])
        if recorded.utcoffset() is None or recorded.utcoffset().total_seconds() != 0:
            raise ValueError("Decision timestamp must use UTC")
        if not isinstance(decision.get("events"), list) or len(decision["events"]) != 2:
            raise ValueError("Record two event decisions, including explicit abstentions if needed")
        for event in decision["events"]:
            if (not isinstance(event.get("query"), str) or not normalized(event["query"])
                    or not isinstance(event.get("note"), str) or not event["note"].strip()):
                raise ValueError("Each event needs a query and visual inspection note")
            if event.get("decision") in ("reject", "abstain"):
                continue
            if event.get("decision") != "accept" or event.get("unique_occurrence") is not True:
                raise ValueError("Acceptance requires a confirmed unique occurrence")
            window = event.get("interval")
            if not valid_window(window, packet["duration"]) or window == [0, packet["duration"]]:
                raise ValueError("Expected one localized interval in seconds")
            confidence = event.get("confidence")
            if type(confidence) not in (float, int) or not 0 <= confidence <= 1:
                raise ValueError("Confidence must be a number between 0 and 1")
            evidence = event.get("evidence")
            if not isinstance(evidence, dict) or set(evidence) != {"before", "inside", "after"}:
                raise ValueError("Require before/inside/after evidence")
            start, end = window
            for group, points in evidence.items():
                if not isinstance(points, list) or not points:
                    raise ValueError(f"Missing frame references for {group}")
                for stamp in points:
                    if type(stamp) not in (int, float) or not any(
                        abs(stamp - t) < 1e-6 for t in packet["timestamps"]
                    ):
                        raise ValueError("Reference actual packet timestamps")
                    if ((group == "inside" and not start <= stamp <= end)
                        or (group == "before" and stamp >= start and start != 0)
                        or (group == "after" and stamp <= end and end != packet["duration"])):
                        raise ValueError(f"Frame falls outside its evidence phase: {group}")
            if (start > 0 and start - max(evidence["before"]) > 2 * packet["sampling_seconds"]
                    or end < packet["duration"] and min(evidence["after"]) - end > 2 * packet["sampling_seconds"]):
                raise ValueError("Boundary evidence must be near the event transition")

    def check(self, vid, attempt, role, decision):
        case = self.case(vid, attempt)
        path = case / role / "task.json"
        packet = read_json(path)
        for name, expected in packet["sheets"].items():
            require_hash(path.parent / name, expected)
        for ref in decision.get("boundary_packets", []):
            if not re.fullmatch(r"\d+\.\d{6}_\d+\.\d{6}", ref):
                raise ValueError("Invalid boundary evidence reference")
            folder = path.parent / "boundary" / ref
            boundary = read_json(folder / "evidence.json")
            if boundary["task_sha256"] != digest(path):
                raise ValueError("Boundary evidence belongs to a different task")
            for name, expected in boundary["sheets"].items():
                require_hash(folder / name, expected)
            packet["timestamps"].extend(boundary["timestamps"])
        self.validate_decision(decision, packet | {"task_sha256": digest(path)})
        if role == "review":
            if [e["query"] for e in decision["events"]] != packet["queries"]:
                raise ValueError("Review must evaluate the exact candidate query in each slot")
            generation = read_json(case / "generation/decision.json")
            if generation["session"] == decision["session"]:
                raise ValueError("Use a separate blind AI reasoning session")
            if datetime.fromisoformat(decision["timestamp_utc"]) < datetime.fromisoformat(generation["timestamp_utc"]):
                raise ValueError("Review timestamp predates generation")

    def record(self, vid, attempt, role, decision):
        self.require_open()
        if role not in ("generation", "review"):
            raise ValueError("Unknown decision role")
        self.check(vid, attempt, role, decision)
        save_json(self.case(vid, attempt) / role / "decision.json", decision)

    def retire(self, vid, attempt, decision):
        self.require_open()
        case = self.case(vid, attempt)
        if not (case / "review/decision.json").is_file():
            raise ValueError("Retirement requires a recorded review")
        if not all(isinstance(decision.get(k), str) and decision[k].strip()
                   for k in ("agent", "reason")):
            raise ValueError("Retirement requires a reviewer identity and reason")
        save_json(case / "retired.json", decision)

    def accepted(self, threshold=0.70):
        if not 0 < threshold <= 1:
            raise ValueError("Agreement threshold must be in (0, 1]")
        rows, sidecar = [], []
        states = {}
        for vid, original in sorted(self.parent.test.items()):
            attempts = sorted((self.work / "cases" / vid).glob("[0-9][0-9][0-9]"))
            history = []
            states[vid] = {"status": "UNRESOLVED", "attempts": history}
            accepted = False
            for case in attempts:
                attempt = int(case.name)
                gen = case / "generation/decision.json"
                rev = case / "review/decision.json"
                entry = {"attempt": attempt, "status": "WAITING_GENERATION"}
                history.append(entry)
                if not gen.is_file():
                    continue
                generation = read_json(gen)
                self.check(vid, attempt, "generation", generation)
                a = generation["events"]
                entry["generation_decisions"] = [x["decision"] for x in a]
                if not all(x["decision"] == "accept" for x in a):
                    entry["status"] = "GENERATION_REJECTED"
                    continue
                if not rev.is_file():
                    entry["status"] = "WAITING_REVIEW"
                    continue
                review = read_json(rev)
                self.check(vid, attempt, "review", review)
                entry["review_decisions"] = [x["decision"] for x in review["events"]]
                if (case / "retired.json").exists():
                    entry.update(status="RETIRED", retirement=read_json(case / "retired.json"))
                    continue
                b = review["events"]
                if not all(x["decision"] == "accept" for x in b):
                    entry["status"] = "REVIEW_REJECTED"
                    continue
                agreements = [tiou(g["interval"], r["interval"]) for g, r in zip(a, b)]
                entry["tiou"] = agreements
                if min(agreements) < threshold:
                    entry["status"] = "BOUNDARY_DISAGREEMENT"
                    continue
                if len({normalized(x) for x in [original["query"], *(e["query"] for e in a)]}) != 3:
                    entry["status"] = "DUPLICATE_QUERY"
                    continue
                if accepted:
                    entry["status"] = "SUPERSEDED_BY_EARLIER_ACCEPTED_ATTEMPT"
                    continue
                for slot, (g, r, agreement) in enumerate(zip(a, b, agreements), 1):
                    qid = f"d1500_test450_ai_v1_{vid}_{slot}"
                    rows.append({
                        "qid": qid, "vid": vid, "query": g["query"], "duration": original["duration"],
                        "relevant_windows": [g["interval"]], "saliency_scores": None, "relevant_clip_ids": None,
                    })
                    sidecar.append({
                        "qid": qid, "vid": vid, "split": "test", "is_original_query": False,
                        "annotation_source": SOURCE, "annotation_version": VERSION,
                        "parent_dataset_version": "lumae_ads_d1500", "video_sha256": original["metadata"]["source_video_sha256"],
                        "method": "timestamped visual inspection and blind AI review",
                        "generation": {**generation, "events": [g], "decision_sha256": digest(gen)},
                        "review": {**review, "events": [r], "decision_sha256": digest(rev)},
                        "reviewer_agreement_tiou": agreement,
                        "evidence_reference": case.relative_to(self.work).as_posix(),
                        "retry_count": sum("generation_decisions" in prior for prior in history) - 1,
                        "verification_status": "ACCEPTED_VISUAL_REVIEW",
                        "final_interval": g["interval"], "event_duration": g["interval"][1] - g["interval"][0],
                    })
                accepted = True
                states[vid]["status"] = entry["status"] = "ACCEPTED_VISUAL_REVIEW"
        if set(row["qid"] for row in rows) & self.parent.qids:
            raise ValueError("New QID collides with the frozen parent")
        return rows, sidecar, states


def progress(rows, sidecar, states):
    import statistics

    agreements = [r["reviewer_agreement_tiou"] for r in sidecar]
    confidence = [r["generation"]["events"][0]["confidence"] for r in sidecar]
    review_confidence = [r["review"]["events"][0]["confidence"] for r in sidecar]
    attempts = [a for state in states.values() for a in state["attempts"]]
    return {
        "visually_accepted_queries": len(rows),
        "unresolved_videos": sum(s["status"] == "UNRESOLVED" for s in states.values()),
        "videos_with_attempts": sum(bool(s["attempts"]) for s in states.values()),
        "attempt_outcomes": dict(Counter(a["status"] for a in attempts)),
        "generation_decisions": dict(Counter(d for a in attempts for d in a.get("generation_decisions", []))),
        "review_decisions": dict(Counter(d for a in attempts for d in a.get("review_decisions", []))),
        "retry_attempts": sum(max(0, sum("generation_decisions" in a for a in s["attempts"]) - 1)
                              for s in states.values()),
        "mean_agreement_tiou": statistics.mean(agreements) if agreements else None,
        "mean_generation_confidence": statistics.mean(confidence) if confidence else None,
        "mean_review_confidence": statistics.mean(review_confidence) if review_confidence else None,
        "per_video": states,
    }
