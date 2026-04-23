from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install ultralytics in the SageMaker model image (see src/requirements.txt).") from exc


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_fn(model_dir: str):
    mp = Path(model_dir)
    weights = mp / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing detector weights at {weights}")
    metadata = _load_json(mp / "metadata.json", {})
    model = YOLO(str(weights))
    return {"model": model, "metadata": metadata}


def input_fn(input_data: str, content_type: str):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(input_data)
    if isinstance(payload, dict) and "instances" in payload:
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    return [payload]


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def predict_fn(instances, model_artifacts):
    model = model_artifacts["model"]
    meta = model_artifacts["metadata"]
    imgsz = int(meta.get("imgsz", 960))

    preds = []
    for item in instances:
        if "image_b64" not in item:
            raise ValueError("Each instance must include image_b64.")
        img = _decode_image(item["image_b64"])
        conf = float(item.get("conf", meta.get("default_detect_conf", 0.25)))
        iou = float(item.get("iou", meta.get("default_detect_iou", 0.45)))
        results = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        r = results[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for j in range(len(r.boxes)):
                b = r.boxes[j]
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0].item())
                dets.append(
                    {
                        "xyxy": xyxy,
                        "confidence": float(b.conf[0].item()),
                        "class_index": cls_id,
                        "label": str(model.names.get(cls_id, "card")),
                    }
                )
        preds.append({"detections": dets})
    return {"predictions": preds}


def output_fn(prediction, accept: str):
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction), "application/json"
