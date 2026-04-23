from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_eval_transform(img_size: int, mean: list[float], std: list[float]):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def model_fn(model_dir: str):
    model_path = Path(model_dir)
    classes = _load_json(model_path / "classes.json", [])
    metadata = _load_json(model_path / "metadata.json", {})

    img_size = int(metadata.get("img_size", 224))
    mean = metadata.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = metadata.get("imagenet_std", [0.229, 0.224, 0.225])

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

    state = torch.load(model_path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "classes": classes,
        "transform": _build_eval_transform(img_size, mean, std),
    }


def input_fn(input_data: str, content_type: str):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(input_data)
    if isinstance(payload, dict) and "instances" in payload:
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    return [payload]


def _decode_image(image_b64: str):
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


def predict_fn(instances, model_artifacts):
    model = model_artifacts["model"]
    classes = model_artifacts["classes"]
    tfm = model_artifacts["transform"]

    images = []
    top_ks = []
    for item in instances:
        if "image_b64" not in item:
            raise ValueError("Each instance must include image_b64.")
        images.append(tfm(_decode_image(item["image_b64"])))
        top_ks.append(int(item.get("top_k", 3)))

    x = torch.stack(images, dim=0)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    preds = []
    for i in range(probs.shape[0]):
        k = max(1, min(top_ks[i], probs.shape[1]))
        values, indices = torch.topk(probs[i], k=k)
        top_preds = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            label = classes[idx] if 0 <= idx < len(classes) else str(idx)
            top_preds.append({"label": label, "probability": float(score), "class_index": int(idx)})
        preds.append(
            {
                "top_prediction": top_preds[0],
                "top_k": top_preds,
            }
        )
    return {"predictions": preds}


def output_fn(prediction, accept: str):
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction), "application/json"
