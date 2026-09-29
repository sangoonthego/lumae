from pathlib import Path
import csv
import hashlib

ROOT = Path("local_data/annotations/lumae_ads")

files = [
    ROOT / "pilot_annotations.csv",
    ROOT / "pilot_annotations_secondary.csv",
]

for path in files:
    print("\n" + "=" * 80)
    print("FILE:", path)

    raw = path.read_bytes()
    print("SHA256:", hashlib.sha256(raw).hexdigest())

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print("ROWS:", len(rows))

    if rows:
        print("COLUMNS:")
        for col in rows[0].keys():
            print(" -", col)

        print("\nFIRST 3 SAMPLES:")
        for row in rows[:3]:
            print({
                "sample_id": row.get("sample_id"),
                "video_filename": row.get("video_filename"),
                "query": row.get("query"),
                "gt_start_seconds": row.get("gt_start_seconds"),
                "gt_end_seconds": row.get("gt_end_seconds"),
                "annotator_id": row.get("annotator_id"),
                "review_status": row.get("review_status"),
                "annotation_notes": row.get("annotation_notes"),
            })