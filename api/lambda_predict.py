from __future__ import annotations

import base64
import json
import os

import boto3

sagemaker_runtime = boto3.client("sagemaker-runtime")
ssm = boto3.client("ssm")


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


def _parse_event(event: dict) -> dict:
    body = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _active_endpoint_name() -> str:
    explicit = os.environ.get("SAGEMAKER_ENDPOINT", "").strip()
    if explicit:
        return explicit

    param_name = os.environ.get("ACTIVE_ENDPOINT_PARAM", "/blackjack/active-endpoint")
    value = ssm.get_parameter(Name=param_name)["Parameter"]["Value"].strip()
    if not value:
        raise RuntimeError(f"SSM parameter {param_name} is empty.")
    return value


def lambda_handler(event, _context):
    try:
        if event.get("httpMethod") == "OPTIONS":
            return _response(200, {"ok": True})

        endpoint_name = _active_endpoint_name()

        req = _parse_event(event)
        image_b64 = req.get("image_base64")
        if not image_b64:
            return _response(400, {"error": "image_base64 is required"})

        payload = {
            "instances": [
                {
                    "image_b64": image_b64,
                    "top_k": int(req.get("top_k", 3)),
                }
            ]
        }

        sm_resp = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )
        result = json.loads(sm_resp["Body"].read().decode("utf-8"))
        return _response(200, {"endpoint": endpoint_name, **result})
    except Exception as exc:
        return _response(500, {"error": str(exc)})
