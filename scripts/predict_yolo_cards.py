from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to trained YOLO weights, e.g. best.pt")
    p.add_argument("--source", required=True, help="Image/video path, folder, or webcam index")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="yolo_runs")
    p.add_argument("--name", default="predict-cards")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        verbose=False,
    )

    serializable = []
    for frame_idx, r in enumerate(results):
        frame_items = []
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            frame_items.append(
                {
                    "frame_index": frame_idx,
                    "class_id": cls_id,
                    "class_name": model.names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "xyxy": [float(v) for v in xyxy],
                }
            )
        serializable.extend(frame_items)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Detections written to {out_path}")
    print(f"Total detections: {len(serializable)}")


if __name__ == "__main__":
    main()
