from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import random
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "ultralytics is required for detector training. Install it in the training image."
    ) from exc


def _extract_sm_context() -> dict:
    ctx = {
        "training_job_name": os.environ.get("TRAINING_JOB_NAME", ""),
        "current_host": os.environ.get("SM_CURRENT_HOST", ""),
        "hosts": os.environ.get("SM_HOSTS", ""),
        "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE", ""),
        "num_gpus": os.environ.get("SM_NUM_GPUS", ""),
        "num_cpus": os.environ.get("SM_NUM_CPUS", ""),
    }
    raw = os.environ.get("SM_TRAINING_ENV")
    if raw:
        try:
            env_obj = json.loads(raw)
            ctx["training_job_name"] = ctx["training_job_name"] or env_obj.get("job_name", "")
            ctx["current_host"] = ctx["current_host"] or env_obj.get("current_host", "")
            if not ctx["hosts"] and "hosts" in env_obj:
                ctx["hosts"] = env_obj["hosts"]
            if not ctx["instance_type"] and "current_instance_type" in env_obj:
                ctx["instance_type"] = env_obj["current_instance_type"]
            if not ctx["num_gpus"] and "num_gpus" in env_obj:
                ctx["num_gpus"] = env_obj["num_gpus"]
            if not ctx["num_cpus"] and "num_cpus" in env_obj:
                ctx["num_cpus"] = env_obj["num_cpus"]
        except json.JSONDecodeError:
            pass
    return ctx


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    no_scheme = uri[5:]
    if "/" not in no_scheme:
        raise ValueError(f"S3 URI must include key path, got: {uri}")
    bucket, key = no_scheme.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI, got: {uri}")
    return bucket, key


def _append_metrics_csv_to_s3(s3_uri: str, row: dict) -> None:
    if not s3_uri:
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as e:  # pragma: no cover
        print(f"WARN could not import boto3/botocore for metrics CSV upload: {e}")
        return

    try:
        bucket, key = _parse_s3_uri(s3_uri)
    except ValueError as e:
        print(f"WARN metrics CSV upload skipped: {e}")
        return

    s3 = boto3.client("s3")
    existing_rows = []
    fieldnames = list(row.keys())
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(body))
        existing_rows = list(reader)
        if reader.fieldnames:
            merged = list(reader.fieldnames)
            for fn in fieldnames:
                if fn not in merged:
                    merged.append(fn)
            fieldnames = merged
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404"):
            print(f"WARN failed to read existing metrics CSV {s3_uri}: {e}")
            return

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in existing_rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    writer.writerow({k: row.get(k, "") for k in fieldnames})

    try:
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
        print(f"wrote run-metrics CSV row to {s3_uri}")
    except Exception as e:
        print(f"WARN failed to upload metrics CSV to {s3_uri}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO card detector in SageMaker script mode.")

    p.add_argument("--data-root", "--data_root", default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    p.add_argument(
        "--unpack-dir",
        "--unpack_dir",
        default=os.environ.get("SM_CHANNEL_UNPACK", "/opt/ml/input/data/unpacked"),
        help="Where to unzip detection dataset when channel contains a zip.",
    )
    p.add_argument(
        "--work-dir",
        "--work_dir",
        default="/opt/ml/input/data/processed",
        help="Where to write converted single-class YOLO dataset.",
    )
    p.add_argument(
        "--model-dir",
        "--model_dir",
        default=os.environ.get("SM_MODEL_DIR", str(Path("outputs") / "model")),
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", str(Path("outputs"))),
    )
    p.add_argument(
        "--metrics-s3-uri",
        "--metrics_s3_uri",
        default="",
        help="s3://bucket/path.csv to append one row of run metrics per training job.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-fraction", "--sample_fraction", type=float, default=1.0)
    p.add_argument("--base-model", "--base_model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--close-mosaic", "--close_mosaic", type=int, default=10)
    p.add_argument("--degrees", type=float, default=15.0)
    p.add_argument("--translate", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--perspective", type=float, default=0.0005)
    p.add_argument("--hsv-h", "--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv-s", "--hsv_s", type=float, default=0.5)
    p.add_argument("--hsv-v", "--hsv_v", type=float, default=0.35)
    p.add_argument("--project-name", "--project_name", default="cards-yolov8n")

    args = p.parse_args()

    def _hp(name: str, cast, fallback):
        v = os.environ.get(f"SM_HP_{name.upper().replace('-', '_')}")
        if v is None or v == "":
            return fallback
        return cast(v)

    args.seed = _hp("seed", int, args.seed)
    args.sample_fraction = _hp("sample_fraction", float, args.sample_fraction)
    args.base_model = _hp("base_model", str, args.base_model)
    args.epochs = _hp("epochs", int, args.epochs)
    args.imgsz = _hp("imgsz", int, args.imgsz)
    args.batch = _hp("batch", int, args.batch)
    args.device = _hp("device", str, args.device)
    args.patience = _hp("patience", int, args.patience)
    args.close_mosaic = _hp("close_mosaic", int, args.close_mosaic)
    args.degrees = _hp("degrees", float, args.degrees)
    args.translate = _hp("translate", float, args.translate)
    args.scale = _hp("scale", float, args.scale)
    args.perspective = _hp("perspective", float, args.perspective)
    args.hsv_h = _hp("hsv_h", float, args.hsv_h)
    args.hsv_s = _hp("hsv_s", float, args.hsv_s)
    args.hsv_v = _hp("hsv_v", float, args.hsv_v)
    args.metrics_s3_uri = _hp("metrics_s3_uri", str, args.metrics_s3_uri)
    args.project_name = _hp("project_name", str, args.project_name)
    return args


def _find_zip(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() == ".zip":
        return root
    if root.is_dir():
        zips = sorted(root.glob("*.zip"))
        if len(zips) == 1:
            return zips[0]
        for name in ("cardsdetection.zip", "cards.zip", "dataset.zip"):
            cand = root / name
            if cand.is_file():
                return cand
    return None


def _extract_detection_root(data_root: Path, unpack_dir: Path) -> Path:
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    if data_root.is_dir():
        ymls = list(data_root.rglob("*.yaml")) + list(data_root.rglob("*.yml"))
        if ymls:
            return data_root

    zpath = _find_zip(data_root)
    if zpath is None:
        raise SystemExit(f"Could not find detection yaml or zip under {data_root}.")

    unpack_root = Path(unpack_dir)
    marker = unpack_root / ".detector_unpacked_marker.txt"
    if marker.is_file():
        return unpack_root

    if unpack_root.exists():
        shutil.rmtree(unpack_root)
    unpack_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(unpack_root)
    marker.write_text(f"unpacked_from={zpath}\n", encoding="utf-8")
    return unpack_root


def _select_source_yaml(root: Path) -> Path:
    ymls = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}])
    if not ymls:
        raise SystemExit(f"No yaml files found under: {root}")
    for name in ("data.yaml", "data.yml", "kaggle_data.yaml", "kaggle_data.yml"):
        for p in ymls:
            if p.name.lower() == name:
                return p
    return ymls[0]


def _clean_rel_path(rel: str | Path) -> Path:
    rel_s = str(rel).replace("\\", "/").strip()
    while rel_s.startswith("../"):
        rel_s = rel_s[3:]
    while rel_s.startswith("./"):
        rel_s = rel_s[2:]
    return Path(rel_s)


def _resolve_split_path(cfg: dict, yaml_path: Path, extraction_root: Path, split: str) -> Path | None:
    rel = cfg.get(split)
    if rel is None and split == "val":
        rel = cfg.get("valid")
    if rel is None and split == "valid":
        rel = cfg.get("val")
    if rel is None:
        return None

    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p

    candidates = []
    yaml_root = yaml_path.parent
    candidates.append((yaml_root / p).resolve())

    if cfg.get("path"):
        base = Path(cfg["path"])
        if not base.is_absolute():
            base = (yaml_root / base).resolve()
        candidates.append((base / p).resolve())

    cleaned = _clean_rel_path(p)
    candidates.append((extraction_root / cleaned).resolve())

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def _resolve_images_dir(split_root: Path | None) -> Path | None:
    if split_root is None:
        return None
    if (split_root / "images").exists():
        return split_root / "images"
    return split_root


def _resolve_labels_dir(split_root: Path, images_dir: Path) -> Path:
    candidates = [
        split_root / "labels",
        images_dir.parent / "labels",
        Path(str(images_dir).replace("/images", "/labels")),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.glob("*") if p.suffix.lower() in exts])


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_results_csv(save_dir: Path) -> list[dict]:
    path = save_dir / "results.csv"
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _best_epoch_from_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    score_cols = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)")
    best_idx = 0
    best_score = -1.0
    for i, row in enumerate(rows):
        score = 0.0
        for col in score_cols:
            if col in row and row[col] != "":
                score = _to_float(row[col], 0.0)
                break
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx + 1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    extraction_root = _extract_detection_root(Path(args.data_root), Path(args.unpack_dir))
    source_yaml = _select_source_yaml(extraction_root)
    with open(source_yaml, "r", encoding="utf-8") as f:
        src_cfg = json.loads(json.dumps(yaml.safe_load(f)))

    out_data_root = work_dir / "cards_binary_detection"
    if out_data_root.exists():
        shutil.rmtree(out_data_root)
    out_data_root.mkdir(parents=True, exist_ok=True)

    stats = {}
    resolved_split_paths = {}
    label_dirs = {}
    split_map = [("train", "train"), ("val", "val"), ("test", "test")]
    for split_key, out_split in split_map:
        split_root = _resolve_split_path(src_cfg, source_yaml, extraction_root, split_key)
        resolved_split_paths[split_key] = str(split_root) if split_root is not None else None
        if split_root is None or not split_root.exists():
            continue

        src_images = _resolve_images_dir(split_root)
        if src_images is None or not src_images.exists():
            continue
        src_labels = _resolve_labels_dir(split_root, src_images)
        label_dirs[split_key] = str(src_labels)

        out_images = out_data_root / "images" / out_split
        out_labels = out_data_root / "labels" / out_split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        imgs = _list_images(src_images)
        if args.sample_fraction < 1.0:
            k = max(1, int(len(imgs) * args.sample_fraction))
            imgs = sorted(random.sample(imgs, k))

        kept_boxes = 0
        with_labels = 0
        missing_labels = 0
        for img_path in imgs:
            shutil.copy(img_path, out_images / img_path.name)
            src_label = src_labels / f"{img_path.stem}.txt"
            out_label = out_labels / f"{img_path.stem}.txt"

            if not src_label.exists():
                missing_labels += 1
                out_label.write_text("", encoding="utf-8")
                continue

            out_lines = []
            for line in src_label.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                out_lines.append("0 " + " ".join(parts[1:]))
                kept_boxes += 1

            if out_lines:
                with_labels += 1
            out_label.write_text("\n".join(out_lines), encoding="utf-8")

        stats[out_split] = {
            "images": len(imgs),
            "images_with_labels": with_labels,
            "missing_label_files": missing_labels,
            "boxes": kept_boxes,
        }

    if "train" not in stats or "val" not in stats:
        raise SystemExit(
            "Converted dataset is missing train/val splits. "
            f"Resolved paths: {json.dumps(resolved_split_paths, indent=2)}"
        )

    total_boxes = sum(v["boxes"] for v in stats.values())
    if total_boxes == 0:
        raise SystemExit(f"Converted dataset has zero boxes. Label dirs used: {label_dirs}")

    binary_yaml = {
        "path": str(out_data_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if "test" in stats else "images/val",
        "names": {0: "card"},
    }
    binary_yaml_path = out_data_root / "data_binary.yaml"
    with open(binary_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(binary_yaml, f, sort_keys=False)

    run_project = out_dir / "yolo_runs"
    model = YOLO(args.base_model)
    t0 = time.time()
    train_results = model.train(
        data=str(binary_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        project=str(run_project),
        name=args.project_name,
    )
    train_seconds = float(time.time() - t0)

    save_dir = Path(getattr(train_results, "save_dir", run_project / args.project_name))
    best_path = Path(model.trainer.best) if getattr(model, "trainer", None) else save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise SystemExit(f"Training completed but best checkpoint was not found at: {best_path}")

    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(data=str(binary_yaml_path), imgsz=args.imgsz, device=args.device, split="val")
    test_metrics = None
    if "test" in stats:
        test_metrics = best_model.val(data=str(binary_yaml_path), imgsz=args.imgsz, device=args.device, split="test")

    val_map = _to_float(getattr(val_metrics.box, "map", 0.0))
    val_map50 = _to_float(getattr(val_metrics.box, "map50", 0.0))
    val_precision = _to_float(getattr(val_metrics.box, "mp", 0.0))
    val_recall = _to_float(getattr(val_metrics.box, "mr", 0.0))
    val_map75 = _to_float(getattr(val_metrics.box, "map75", 0.0))

    test_map = _to_float(getattr(test_metrics.box, "map", 0.0)) if test_metrics else 0.0
    test_map50 = _to_float(getattr(test_metrics.box, "map50", 0.0)) if test_metrics else 0.0
    test_precision = _to_float(getattr(test_metrics.box, "mp", 0.0)) if test_metrics else 0.0
    test_recall = _to_float(getattr(test_metrics.box, "mr", 0.0)) if test_metrics else 0.0
    test_map75 = _to_float(getattr(test_metrics.box, "map75", 0.0)) if test_metrics else 0.0

    rows = _read_results_csv(save_dir)
    epochs_ran = len(rows) if rows else args.epochs
    best_epoch_1idx = _best_epoch_from_rows(rows)
    stopped_early = epochs_ran < args.epochs
    final_train_box_loss = _to_float(rows[-1].get("train/box_loss", 0.0), 0.0) if rows else 0.0
    final_val_box_loss = _to_float(rows[-1].get("val/box_loss", 0.0), 0.0) if rows else 0.0

    print(f"METRIC val_map50_95 = {val_map:.6f}")
    print(f"METRIC val_map50 = {val_map50:.6f}")
    print(f"METRIC val_map75 = {val_map75:.6f}")
    print(f"METRIC val_precision = {val_precision:.6f}")
    print(f"METRIC val_recall = {val_recall:.6f}")
    print(f"METRIC test_map50_95 = {test_map:.6f}")
    print(f"METRIC test_map50 = {test_map50:.6f}")
    print(f"METRIC test_map75 = {test_map75:.6f}")
    print(f"METRIC test_precision = {test_precision:.6f}")
    print(f"METRIC test_recall = {test_recall:.6f}")
    print(f"METRIC final_train_box_loss = {final_train_box_loss:.6f}")
    print(f"METRIC final_val_box_loss = {final_val_box_loss:.6f}")
    print(f"METRIC train_seconds = {train_seconds:.3f}")
    print(f"METRIC epochs_ran = {float(epochs_ran)}")

    shutil.copy2(best_path, model_dir / "best.pt")
    if (save_dir / "weights" / "last.pt").exists():
        shutil.copy2(save_dir / "weights" / "last.pt", model_dir / "last.pt")
    with open(model_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(["card"], f, indent=2)
    with open(model_dir / "data_binary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(binary_yaml, f, sort_keys=False)

    sm_ctx = _extract_sm_context()
    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "yolov8n",
        "base_model": args.base_model,
        "num_classes": 1,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs_requested": args.epochs,
        "epochs_ran": epochs_ran,
        "best_epoch_1idx": best_epoch_1idx,
        "early_stopped": stopped_early,
        "patience": args.patience,
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "close_mosaic": args.close_mosaic,
        "augment": {
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "perspective": args.perspective,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
        },
        "dataset_stats": stats,
        "resolved_split_paths": resolved_split_paths,
        "resolved_label_dirs": label_dirs,
        "source_yaml": str(source_yaml),
        "binary_yaml_path": str(binary_yaml_path),
        "train_seconds": train_seconds,
        "val_metrics": {
            "map50_95": val_map,
            "map50": val_map50,
            "map75": val_map75,
            "precision": val_precision,
            "recall": val_recall,
        },
        "test_metrics": {
            "map50_95": test_map,
            "map50": test_map50,
            "map75": test_map75,
            "precision": test_precision,
            "recall": test_recall,
        },
        "final_metrics": {
            "final_train_box_loss": final_train_box_loss,
            "final_val_box_loss": final_val_box_loss,
        },
        "runtime_context": sm_ctx,
        "metrics_s3_uri": args.metrics_s3_uri,
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_path),
        "device": args.device,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if save_dir.exists():
        archived_run_dir = out_dir / "yolo_training_run"
        if archived_run_dir.exists():
            shutil.rmtree(archived_run_dir)
        shutil.copytree(save_dir, archived_run_dir)

    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir / "best.pt", arcname="best.pt")
        if (model_dir / "last.pt").exists():
            tar.add(model_dir / "last.pt", arcname="last.pt")
        tar.add(model_dir / "classes.json", arcname="classes.json")
        tar.add(model_dir / "metadata.json", arcname="metadata.json")
        tar.add(model_dir / "data_binary.yaml", arcname="data_binary.yaml")

    run_metrics_row = {
        "created_utc": meta["created_utc"],
        "training_job_name": sm_ctx.get("training_job_name", ""),
        "model": meta["model"],
        "base_model": args.base_model,
        "epochs_ran": epochs_ran,
        "best_epoch_1idx": best_epoch_1idx,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "sample_fraction": args.sample_fraction,
        "val_map50_95": val_map,
        "val_map50": val_map50,
        "val_map75": val_map75,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_map50_95": test_map,
        "test_map50": test_map50,
        "test_map75": test_map75,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "train_seconds": train_seconds,
        "num_train_images": stats.get("train", {}).get("images", 0),
        "num_val_images": stats.get("val", {}).get("images", 0),
        "num_test_images": stats.get("test", {}).get("images", 0),
        "train_boxes": stats.get("train", {}).get("boxes", 0),
        "val_boxes": stats.get("val", {}).get("boxes", 0),
        "test_boxes": stats.get("test", {}).get("boxes", 0),
        "instance_type": sm_ctx.get("instance_type", ""),
        "device": args.device,
        "model_artifact_local": str(tar_path),
    }
    local_metrics_csv = out_dir / "run_metrics.csv"
    with open(local_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_metrics_row.keys()))
        writer.writeheader()
        writer.writerow(run_metrics_row)
    print(f"wrote {local_metrics_csv}")

    if args.metrics_s3_uri:
        _append_metrics_csv_to_s3(args.metrics_s3_uri, run_metrics_row)

    print(f"wrote {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
