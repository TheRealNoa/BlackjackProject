from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/cards_detection.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project", default="yolo_runs")
    p.add_argument("--name", default="cards-yolov8n")
    p.add_argument("--degrees", type=float, default=15.0)
    p.add_argument("--translate", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--perspective", type=float, default=0.0005)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.5)
    p.add_argument("--hsv-v", type=float, default=0.35)
    p.add_argument("--close-mosaic-epoch", type=int, default=10)
    return p.parse_args()


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        close_mosaic=args.close_mosaic_epoch,
    )

    best_path = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    if not best_path or not best_path.exists():
        raise SystemExit("Training completed but best checkpoint was not found.")

    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": args.data,
        "base_model": args.model,
        "best_checkpoint": str(best_path),
        "results_dir": str(getattr(train_results, "save_dir", "")),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "metrics": {
            "mAP50-95": _to_float(getattr(val_metrics.box, "map", 0.0)),
            "mAP50": _to_float(getattr(val_metrics.box, "map50", 0.0)),
            "precision": _to_float(getattr(val_metrics.box, "mp", 0.0)),
            "recall": _to_float(getattr(val_metrics.box, "mr", 0.0)),
        },
    }

    out_dir = Path(train_results.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
