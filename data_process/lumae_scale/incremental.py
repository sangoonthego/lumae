"""Exact D200 + 300 freeze, preserving the same byte-for-byte benchmark."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

from data_process.annotation.assisted_deployment import ALGORITHM_SHA256,verify_frozen_ranker
from .freeze import accepted_records,canonical_row,validate_annotation,_jsonl_bytes,portable_copy
from .models import atomic_json,digest,load_jsonl,read_json
from .quality_gate import interval_errors
from .telemetry import aggregate


def verify_parent(layout):
    manifest = read_json(layout.base / "dataset_manifest.json")
    canonical = layout.base / f"{layout.base_version}_all.jsonl"
    if manifest["total_samples"] != 200 or digest(canonical) != manifest["dataset_sha256"]:
        raise ValueError("Frozen D200 parent changed or incomplete")
    for line in (layout.base / "checksums.sha256").read_text().splitlines():
        sha,name=line.split("  ",1)
        if digest(layout.base/name)!=sha: raise ValueError("D200 checksum mismatch: "+name)
    rows=load_jsonl(canonical)
    if len(rows)!=200 or len({r["vid"] for r in rows})!=200 or len({r["qid"] for r in rows})!=200:
        raise ValueError("Invalid D200 uniqueness")
    if len({x["sha256"] for x in manifest["source_video_sha256"].values()})!=200:
        raise ValueError("Invalid parent content uniqueness")
    return manifest


class ProtectedInputs:
    def __init__(self,layout):
        self.root=layout.root
        self.manifest_path=layout.root/"local_data/manifests/lumae_ads_d500_protected_inputs.json"
        self.snapshot=read_json(self.manifest_path)["sha256"]
        self.stats={rel:(self.root/rel).stat() for rel in self.snapshot}

    def verify(self,*,full=False):
        for rel,sha in self.snapshot.items():
            p=self.root/rel
            stat=p.stat(); old=self.stats[rel]
            if full or (stat.st_size,stat.st_mtime_ns)!=(old.st_size,old.st_mtime_ns):
                if digest(p)!=sha: raise RuntimeError("Protected D200/research artifact changed: "+rel)
                self.stats[rel]=stat
        return {"status":"PASS","protected_files":len(self.snapshot),"hash_mode":"FULL_SHA256" if full else "stat check with SHA256 on changes"}


def append_ledger(layout,kind,video_id,**details):
    layout.ledger.parent.mkdir(parents=True,exist_ok=True)
    row={"time_utc":datetime.now(timezone.utc).isoformat(),"event":kind,"source_video_id":video_id,**details}
    with layout.ledger.open("a",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")
        stream.flush(); os.fsync(stream.fileno())


def progress(layout,result=None):
    records=accepted_records(layout.accepted)
    state=read_json(layout.selection) if layout.selection.is_file() else {"candidates":[]}
    counts=Counter(r["status"] for r in state["candidates"])
    report={"dataset_version":layout.version,"base_count":200,"new_accepted":len(records),
            "valid_total":200+len(records),"remaining_new":300-len(records),"source_states":dict(counts),
            "terminal_rejection_kinds":dict(Counter(r.get("failure_kind") for r in state["candidates"] if r["status"]=="rejected")),
            "frozen":(layout.dataset/f"{layout.version}_all.jsonl").is_file(),
            "phase":"A","external_phase_b":"D200 training features, same M0 checkpoint, unchanged Colab training recipe; not Phase A blockers"}
    atomic_json(layout.reports/"build_report.json",report)
    return result if result is not None else report


def immutable_write(path:Path,data:bytes):
    if path.exists():
        if path.read_bytes()!=data: raise ValueError("Frozen D500 output would change: "+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    # The atomic writer is also used for JSONL bytes; no partial final file.
    import tempfile
    fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def json_bytes(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()


def validate_extension(layout,records):
    parent=verify_parent(layout)
    if len(records)!=300:raise ValueError(f"D500 requires exactly +300 accepted samples; have {len(records)}")
    base=load_jsonl(layout.base/f"{layout.base_version}_all.jsonl")
    base_ids={r["vid"].removeprefix("lumae_ads_") for r in base}
    ids=[r["source_video_id"] for r in records]
    if len(set(ids))!=300 or set(ids)&base_ids:raise ValueError("D500 duplicate source IDs or D200 overlap")
    queries=[r["query"].strip().casefold() for r in records+base]
    if len(set(queries))!=500:raise ValueError("Duplicate D500 query text")
    hashes=[r["source_video_sha256"] for r in records]+[r["sha256"] for r in parent["source_video_sha256"].values()]
    if len(set(hashes))!=500:raise ValueError("Duplicate source video content")
    for record in records:validate_annotation(record,visual_cache=layout.visual_cache)
    new=[canonical_row(r,f"{layout.version}_{r['source_video_id']}") for r in sorted(records,key=lambda r:r["source_video_id"])]
    all_rows=base+new
    if (len(all_rows),len({r["qid"] for r in all_rows}),len({r["vid"] for r in all_rows}))!=(500,500,500):raise ValueError("D500 uniqueness failure")
    if Counter(r["metadata"]["review_provenance"] for r in all_rows)!={"HUMAN_VERIFIED":18,"AI_PSEUDO_LABELED":482}:raise ValueError("D500 provenance failure")
    if any(r["saliency_scores"] is not None or r["relevant_clip_ids"] is not None for r in all_rows):raise ValueError("Fake saliency")
    return parent,base,new


def freeze_incremental(layout):
    ProtectedInputs(layout).verify(full=True)
    records=accepted_records(layout.accepted)
    parent,base,new=validate_extension(layout,records)
    verify_frozen_ranker(layout.root)
    parent_splits={s:load_jsonl(layout.base/f"{layout.base_version}_{s}.jsonl") for s in ("train","val","test")}
    splits={"train":parent_splits["train"]+new,"val":parent_splits["val"],"test":parent_splits["test"]}
    if tuple(len(splits[s]) for s in ("train","val","test"))!=(440,30,30):raise ValueError("D500 split counts")
    vid_sets={s:{r["vid"] for r in rows} for s,rows in splits.items()}
    if any(vid_sets[a]&vid_sets[b] for a,b in (("train","val"),("train","test"),("val","test"))):raise ValueError("Video leakage")
    if not {r["vid"] for r in new}<=vid_sets["train"]:raise ValueError("New sample outside Train")
    d48={r["vid"] for r in base if not r["qid"].startswith("lumae_ads_d200_")}
    if len(d48)!=48 or not d48<=vid_sets["train"]:raise ValueError("D48 outside Train")
    for name in ("all","train","val","test"):
        old=(layout.base/f"{layout.base_version}_{name}.jsonl").read_bytes()
        if not old.endswith(b"\n"):raise ValueError("Parent JSONL lacks terminal newline")
        data=old+_jsonl_bytes(new) if name in ("all","train") else old
        immutable_write(layout.dataset/f"{layout.version}_{name}.jsonl",data)
    split_manifest={"seed":layout.seed,"counts":{s:len(r) for s,r in splits.items()},
                    "fixed_evaluation_parent":layout.base_version,"new_samples_train_only":True,
                    "d48_train_only":True,"video_ids":{s:sorted(ids) for s,ids in vid_sets.items()},
                    "val_byte_sha256":digest(layout.base/f"{layout.base_version}_val.jsonl"),
                    "test_byte_sha256":digest(layout.base/f"{layout.base_version}_test.jsonl")}
    immutable_write(layout.dataset/"split_manifest.json",json_bytes(split_manifest))
    immutable_write(layout.dataset/"selection_manifest.json",json_bytes(portable_copy(read_json(layout.selection))))
    immutable_write(layout.dataset/"build_ledger.jsonl",_jsonl_bytes(portable_copy(load_jsonl(layout.ledger))))
    audit=[{"qid":r["qid"],"vid":r["vid"],"query":r["query"],"window":r["relevant_windows"][0],
            "review_provenance":r["metadata"]["review_provenance"],"verifier":r["metadata"].get("verifier"),
            "accepted_record_sha256":digest(layout.accepted/(r["metadata"]["source_video_id"]+".json"))} for r in new]
    immutable_write(layout.dataset/"annotation_audit.json",json_bytes(audit))
    manifest={"dataset_version":layout.version,"base_dataset_version":layout.base_version,
              "base_dataset_sha256":parent["dataset_sha256"],"total_samples":500,"reused_d200":200,
              "new_samples":300,"counts":split_manifest["counts"],"seed":layout.seed,
              "dataset_sha256":digest(layout.dataset/f"{layout.version}_all.jsonl"),
              "query_annotation_version":"d200_pipeline_v1","semantic_v3_algorithm_sha256":ALGORITHM_SHA256,
              "source_selection_sha256":digest(layout.dataset/"selection_manifest.json"),
              "split_manifest_sha256":digest(layout.dataset/"split_manifest.json"),
              "provenance_counts":{"human_verified":18,"ai_pseudo_labeled":482},
              "source_video_sha256":{**parent["source_video_sha256"],**{r["qid"]:{"video_filename":r["metadata"]["video_filename"],"sha256":r["metadata"]["source_video_sha256"]} for r in new}},
              "warning":"Val/Test are exactly the fixed D200 AI pseudo labels, not human temporal ground truth."}
    immutable_write(layout.dataset/"dataset_manifest.json",json_bytes(manifest))
    files=sorted(p for p in layout.dataset.iterdir() if p.is_file() and p.name!="checksums.sha256")
    immutable_write(layout.dataset/"checksums.sha256","".join(f"{digest(p)}  {p.name}\n" for p in files).encode())
    handoff={"dataset_version":layout.version,"phase":"A_COMPLETE_PHASE_B_EXTERNAL",
             "total_samples":500,"train":440,"val":30,"test":30,
             "human_verified":18,"ai_pseudo_labeled":482,"dataset_sha256":manifest["dataset_sha256"],
             "dataset_manifest_path":(layout.dataset/"dataset_manifest.json").relative_to(layout.root).as_posix(),
             "dataset_manifest_sha256":digest(layout.dataset/"dataset_manifest.json"),
             "model":"Moment-DETR","feature_extractor":"CLIP ViT-B/32","feature_mode":"CLIP-only",
             "clip_length":2.0,"max_video_length":75,"max_query_length":32,"use_tef":True,
             "video_input_dim":514,"video_clip_storage_dim":512,"text_input_dim":512,
             "has_saliency_gt":False,"saliency_loss_weight":0.0,"seed":layout.seed,
             "initialization_policy":"Same QVHighlights M0 checkpoint SHA as external M1-D200; never continue from M1-D200",
             "m0_checkpoint_status":"EXTERNAL_PHASE_B_TO_BIND_AND_VERIFY",
             "feature_policy":{"reuse_d200_video":200,"reuse_d200_text":200,"encode_new_video":300,"encode_new_text":300,
                               "invalid_parent_cache":"Do not overwrite D200; stop and resolve or rebuild a separate D500 copy with recorded provenance"},
             "feature_manifest_status":"EXTERNAL_PHASE_B_NOT_EXTRACTED",
             "training_recipe_policy":"Reuse externally frozen D200 recipe; Phase A does not implement or alter training",
             "test_policy":"Fixed Test-30 sealed until one explicitly requested final evaluation; never use for selection",
             "evaluation_policy":"Use the externally adopted official Moment-DETR compute_mr_ap/compute_mr_r1 on the full fixed test, plus mIoU and supported best-of-10 diagnostics",
             "annotation_warning":manifest["warning"]}
    for s in ("train","val","test"):
        p=layout.dataset/f"{layout.version}_{s}.jsonl"
        handoff[s+"_manifest"]={"path":p.relative_to(layout.root).as_posix(),"sha256":digest(p)}
    master=write_master(layout,base,new,splits)
    handoff["annotation_master"]={"path":master.relative_to(layout.root).as_posix(),"sha256":digest(master)}
    immutable_write(layout.handoff,json_bytes(handoff))
    atomic_json(layout.reports/"telemetry_summary.json",aggregate(layout.work/"telemetry"))
    progress(layout)
    return manifest


def write_master(layout,base,new,splits):
    mapping={r["qid"]:s for s,rows in splits.items() for r in rows}
    parent=layout.root/"local_data/reports/lumae_ads_d200/lumae_ads_d200_annotation_master.csv"
    if parent.is_file():
        with parent.open(encoding="utf-8",newline="") as stream:
            reader=csv.DictReader(stream); inherited=list(reader);fields=list(reader.fieldnames)
    else:
        fields=["qid","vid","query","start","end","review_provenance","final_split"]
        inherited=[{"qid":r["qid"],"vid":r["vid"],"query":r["query"],"start":r["relevant_windows"][0][0],"end":r["relevant_windows"][0][1],"review_provenance":r["metadata"]["review_provenance"],"final_split":mapping[r["qid"]]} for r in base]
    if len(inherited)!=200:raise ValueError("Parent master CSV must contain 200 rows")
    path=layout.reports/f"{layout.version}_annotation_master.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    with io.StringIO(newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(inherited)
        for row in new:
            m=row["metadata"];vid=m["source_video_id"]
            summary_path=layout.work/"attempts"/(vid+".json")
            t=read_json(summary_path) if summary_path.is_file() else {}
            engineering=m.get("engineering_cache")
            if not engineering and (layout.work/"source_cache"/(vid+".json")).is_file():
                # Early D500 accepted bytes remain immutable after telemetry fields
                # were added; derive their master-CSV cache identity from persisted artifacts.
                from .stage_cache import frame_fingerprint,clip_fingerprint
                index=read_json(layout.visual_cache/vid/"metadata.json")
                review_path=layout.work/"visual_index"/vid/"visual_events.json"
                review=read_json(review_path)
                source=read_json(layout.work/"source_cache"/(vid+".json"))
                engineering={"source":source["identity"],"visual_index":frame_fingerprint(index),
                             "clip":clip_fingerprint(index),"query_review_sha256":digest(review_path),
                             "semantic_refinement":hashlib.sha256(json.dumps(read_json(layout.root/m["semantic_artifact_path"]),sort_keys=True).encode()).hexdigest(),
                             "verifier":m["verifier"].get("cache_identity"),
                             "overview_path":review.get("overview_path"),"overview_sha256":review.get("overview_sha256")}
            engineering=engineering or {}
            values={**m,"qid":row["qid"],"vid":row["vid"],"adsqa_source_id":vid,
                    "query":row["query"],"duration":row["duration"],"start":row["relevant_windows"][0][0],"end":row["relevant_windows"][0][1],
                    "source_sha256":m["source_video_sha256"],"relative_video_path":m["source_video_path"],
                    "final_split":"train","agent_reviewed":True,"semantic_v3_hash":ALGORITHM_SHA256,
                    "router_state":m["router"]["route"],"visual_views_count":t.get("visual_views_count"),
                    "clip_signals":m.get("clip_signals"),"cache_fingerprints":engineering,
                    "fingerprint_scope":"source, visual index, CLIP, reviewed query, Semantic V3/refinement, verifier",
                    "visual_overview_path":engineering.get("overview_path"),
                    "visual_overview_sha256":engineering.get("overview_sha256"),
                    "total_accepted_seconds":t.get("total_ms",0)/1000 if t else None,
                    "total_accepted_minutes":t.get("total_ms",0)/60000 if t else None,
                    "timing_scope":t.get("timing_scope","UNAVAILABLE"),
                    "stages_seconds":{k:(v/1000 if v is not None else None) for k,v in t.get("timings_ms",{}).items()},
                    "pipeline_call_seconds":t.get("pipeline_call_seconds"),"history_state":"accepted",
                    "accepted_record_sha256":digest(layout.accepted/(vid+".json"))}
            writer.writerow({k:json.dumps(values[k],sort_keys=True) if isinstance(values.get(k),(dict,list)) else values.get(k) for k in fields})
        immutable_write(path,stream.getvalue().encode("utf-8"))
    return path


def package_incremental(layout):
    ProtectedInputs(layout).verify(full=True)
    manifest=read_json(layout.dataset/"dataset_manifest.json")
    if manifest["total_samples"]!=500:raise ValueError("Cannot package incomplete D500")
    verify_parent(layout)
    for s in ("val","test"):
        if (layout.dataset/f"{layout.version}_{s}.jsonl").read_bytes() != (layout.base/f"{layout.base_version}_{s}.jsonl").read_bytes():
            raise ValueError("Fixed D200 evaluation split changed before package")
    paths=sorted(p for p in layout.dataset.iterdir() if p.is_file())+[layout.handoff]
    for r in manifest["source_video_sha256"].values():
        p=layout.videos/r["video_filename"]
        if not p.is_file() or digest(p)!=r["sha256"]:raise ValueError("Source video changed before package: "+r["video_filename"])
        paths.append(p)
    master=layout.reports/f"{layout.version}_annotation_master.csv"
    with master.open(encoding="utf-8",newline="") as stream:
        if sum(1 for _ in csv.DictReader(stream))!=500:raise ValueError("D500 annotation master count")
    paths += [master,layout.root/"data_process/lumae_scale/README_D500.md"]
    export=layout.root/"local_data/exports/lumae_ads_d500_colab.zip"
    export.parent.mkdir(parents=True,exist_ok=True)
    partial=export.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for p in paths:archive.write(p,p.relative_to(layout.root).as_posix())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:raise ValueError("ZIP integrity failure")
        names=archive.namelist()
        if len(names)!=len(set(names)) or any(n.startswith(("/","\\")) or ":" in n or ".." in Path(n).parts for n in names):raise ValueError("Nonportable ZIP")
    os.replace(partial,export)
    result={"path":export.relative_to(layout.root).as_posix(),"sha256":digest(export),"bytes":export.stat().st_size,"file_count":len(paths),
            "archive_integrity":"PASS","phase_b_training_artifacts":"EXTERNAL; package contains frozen data/handoff, not a changed training recipe"}
    atomic_json(layout.reports/"colab_export.json",result)
    return result
