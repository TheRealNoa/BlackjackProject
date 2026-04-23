from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import boto3

sagemaker_runtime = boto3.client("sagemaker-runtime")
ssm = boto3.client("ssm")

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required for cropping. Add pillow to your Lambda deployment package "
        "or use the file in api/requirements_orchestrate.txt."
    ) from exc


def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(body),
    }


def _http_method(event: dict) -> str:
    return str(
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method", "POST")
    ).upper()


def _parse_event(event: dict) -> dict:
    body = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _ssm_endpoint(param_name: str) -> str:
    value = ssm.get_parameter(Name=param_name)["Parameter"]["Value"].strip()
    if not value:
        raise RuntimeError(f"SSM parameter {param_name} is empty.")
    return value


def _detector_endpoint() -> str:
    v = os.environ.get("SAGEMAKER_DETECTOR_ENDPOINT", "").strip()
    if v:
        return v
    param = os.environ.get("ACTIVE_DETECTOR_ENDPOINT_PARAM", "/blackjack/active-detector-endpoint")
    return _ssm_endpoint(param)


def _classifier_endpoint() -> str:
    v = os.environ.get("SAGEMAKER_ENDPOINT", "").strip()
    if v:
        return v
    param = os.environ.get("ACTIVE_ENDPOINT_PARAM", "/blackjack/active-endpoint")
    return _ssm_endpoint(param)


def _invoke_json(endpoint: str, payload: dict) -> dict:
    sm_resp = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    return json.loads(sm_resp["Body"].read().decode("utf-8"))


def _expand_xyxy(xyxy: list[float], w: int, h: int, pad_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    px = bw * pad_frac
    py = bh * pad_frac
    nx1 = int(max(0, x1 - px))
    ny1 = int(max(0, y1 - py))
    nx2 = int(min(w - 1, x2 + px))
    ny2 = int(min(h - 1, y2 + py))
    if nx2 <= nx1:
        nx2 = min(w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h - 1, ny1 + 1)
    return nx1, ny1, nx2 + 1, ny2 + 1


def _image_to_jpeg_b64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def lambda_handler(event: dict, _context: Any):
    try:
        if _http_method(event) == "OPTIONS":
            return _response(200, {"ok": True})

        req = _parse_event(event)
        image_b64 = req.get("image_base64")
        if not image_b64:
            return _response(400, {"error": "image_base64 is required"})

        top_k = int(req.get("top_k", 3))
        detect_conf = float(req.get("detect_conf", 0.25))
        detect_iou = float(req.get("detect_iou", 0.45))
        max_detections = int(req.get("max_detections", 12))
        crop_padding = float(req.get("crop_padding", 0.02))
        crop_quality = int(req.get("crop_jpeg_quality", 92))

        detector_ep = _detector_endpoint()
        classifier_ep = _classifier_endpoint()

        det_payload = {
            "instances": [
                {
                    "image_b64": image_b64,
                    "conf": detect_conf,
                    "iou": detect_iou,
                }
            ]
        }
        det_out = _invoke_json(detector_ep, det_payload)
        preds = det_out.get("predictions") or []
        if not preds:
            return _response(500, {"error": "Detector returned no predictions key"})

        dets = preds[0].get("detections") or []
        dets = sorted(dets, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)[: max(0, max_detections)]

        raw = base64.b64decode(image_b64)
        full_img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = full_img.size

        crops_b64: list[str] = []
        crop_meta: list[dict] = []
        for idx, d in enumerate(dets):
            xyxy = d.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            bx1, by1, bx2, by2 = _expand_xyxy([x1, y1, x2, y2], w, h, crop_padding)
            crop = full_img.crop((bx1, by1, bx2, by2))
            crops_b64.append(_image_to_jpeg_b64(crop, quality=crop_quality))
            crop_meta.append(
                {
                    "detection_index": idx,
                    "xyxy": [x1, y1, x2, y2],
                    "crop_xyxy": [bx1, by1, bx2, by2],
                    "detection_confidence": float(d.get("confidence", 0.0)),
                    "label": d.get("label", "card"),
                }
            )

        if not crops_b64:
            return _response(
                200,
                {
                    "detector_endpoint": detector_ep,
                    "classifier_endpoint": classifier_ep,
                    "detections": dets,
                    "cards": [],
                    "message": "No detections above threshold.",
                },
            )

        cls_payload = {"instances": [{"image_b64": b64, "top_k": top_k} for b64 in crops_b64]}
        cls_out = _invoke_json(classifier_ep, cls_payload)
        cls_preds = cls_out.get("predictions") or []

        cards = []
        for i, meta in enumerate(crop_meta):
            card = dict(meta)
            if i < len(cls_preds):
                card["classification"] = cls_preds[i]
            cards.append(card)

        return _response(
            200,
            {
                "detector_endpoint": detector_ep,
                "classifier_endpoint": classifier_ep,
                "detections": dets,
                "cards": cards,
            },
        )
    except Exception as exc:
        return _response(500, {"error": str(exc)})
