from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

import boto3

dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.environ.get("LEADERBOARD_TABLE", "").strip()
INDEX_NAME = os.environ.get("LEADERBOARD_GSI_NAME", "ByWinRate").strip()
GLOBAL_KEY = os.environ.get("LEADERBOARD_GLOBAL_KEY", "GLOBAL").strip()


def _response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,GET,POST",
        },
        "body": json.dumps(body),
    }


def _http_method(event: dict[str, Any]) -> str:
    return str(
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method", "GET")
    ).upper()


def _parse_json_body(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("body") or "{}"
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _query_param(event: dict[str, Any], key: str, default: str) -> str:
    qs = event.get("queryStringParameters") or {}
    value = qs.get(key)
    return str(value) if value is not None else default


def _claims(event: dict[str, Any]) -> dict[str, Any]:
    rc = event.get("requestContext") or {}
    auth = rc.get("authorizer") or {}
    jwt = auth.get("jwt") or {}
    claims = jwt.get("claims")
    return claims if isinstance(claims, dict) else {}


def _to_int(val: Any) -> int:
    if isinstance(val, Decimal):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    return int(str(val or "0"))


def _compute_win_rate_bp(wins: int, games: int) -> int:
    if games <= 0:
        return 0
    return int((wins / games) * 10000)


def _submit_result(event: dict[str, Any]) -> dict[str, Any]:
    if not TABLE_NAME:
        return _response(500, {"error": "LEADERBOARD_TABLE is not configured"})

    claims = _claims(event)
    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        return _response(401, {"error": "Unauthorized: missing user identity claims."})

    body = _parse_json_body(event)
    outcome = str(body.get("outcome") or "").strip().lower()
    if outcome not in {"player", "dealer", "push"}:
        return _response(400, {"error": "outcome must be one of: player, dealer, push"})

    username = str(body.get("username") or claims.get("preferred_username") or "").strip()

    table = dynamodb.Table(TABLE_NAME)
    now_iso = str(event.get("requestContext", {}).get("timeEpoch", ""))

    increments = {"wins": 0, "losses": 0, "pushes": 0, "games_played": 1}
    if outcome == "player":
        increments["wins"] = 1
    elif outcome == "dealer":
        increments["losses"] = 1
    else:
        increments["pushes"] = 1

    update_expr = (
        "ADD wins :wins, losses :losses, pushes :pushes, games_played :games "
        "SET #uname = if_not_exists(#uname, :uname), gsi_pk = :gpk, updated_at = :updated"
    )
    expr_names = {"#uname": "username"}
    expr_values = {
        ":wins": increments["wins"],
        ":losses": increments["losses"],
        ":pushes": increments["pushes"],
        ":games": increments["games_played"],
        ":uname": username or "unknown",
        ":gpk": GLOBAL_KEY,
        ":updated": now_iso,
    }
    if username:
        update_expr = (
            "ADD wins :wins, losses :losses, pushes :pushes, games_played :games "
            "SET #uname = :uname, gsi_pk = :gpk, updated_at = :updated"
        )

    updated = table.update_item(
        Key={"user_id": user_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
        ReturnValues="ALL_NEW",
    )["Attributes"]

    wins = _to_int(updated.get("wins", 0))
    games_played = _to_int(updated.get("games_played", 0))
    win_rate_bp = _compute_win_rate_bp(wins, games_played)
    table.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET win_rate_bp = :bp",
        ExpressionAttributeValues={":bp": win_rate_bp},
    )

    return _response(
        200,
        {
            "ok": True,
            "user_id": user_id,
            "wins": wins,
            "losses": _to_int(updated.get("losses", 0)),
            "pushes": _to_int(updated.get("pushes", 0)),
            "games_played": games_played,
            "win_rate_pct": round(win_rate_bp / 100, 2),
        },
    )


def _get_leaderboard(event: dict[str, Any]) -> dict[str, Any]:
    if not TABLE_NAME:
        return _response(500, {"error": "LEADERBOARD_TABLE is not configured"})

    limit = max(1, min(50, int(_query_param(event, "limit", "10"))))
    min_games = max(1, int(_query_param(event, "min_games", "1")))
    table = dynamodb.Table(TABLE_NAME)

    collected: list[dict[str, Any]] = []
    cursor = None
    while len(collected) < limit:
        kwargs = {
            "IndexName": INDEX_NAME,
            "KeyConditionExpression": "gsi_pk = :gpk",
            "ExpressionAttributeValues": {":gpk": GLOBAL_KEY},
            "ScanIndexForward": False,
            "Limit": 50,
        }
        if cursor:
            kwargs["ExclusiveStartKey"] = cursor
        out = table.query(**kwargs)
        items = out.get("Items") or []
        for item in items:
            games_played = _to_int(item.get("games_played", 0))
            if games_played < min_games:
                continue
            wins = _to_int(item.get("wins", 0))
            losses = _to_int(item.get("losses", 0))
            pushes = _to_int(item.get("pushes", 0))
            win_rate_bp = _to_int(item.get("win_rate_bp", 0))
            collected.append(
                {
                    "user_id": str(item.get("user_id", "")),
                    "username": str(item.get("username", "")),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "games_played": games_played,
                    "win_rate_pct": round(win_rate_bp / 100, 2),
                }
            )
            if len(collected) >= limit:
                break
        cursor = out.get("LastEvaluatedKey")
        if not cursor:
            break

    return _response(200, {"items": collected[:limit]})


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    try:
        method = _http_method(event)
        if method == "OPTIONS":
            return _response(200, {"ok": True})
        if method == "GET":
            return _get_leaderboard(event)
        if method == "POST":
            return _submit_result(event)
        return _response(405, {"error": f"Method {method} not allowed"})
    except Exception as exc:  # pragma: no cover
        return _response(500, {"error": str(exc)})
