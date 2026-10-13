"""Tests for POST /recommendations/events."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_events_requires_auth(client: AsyncClient):
    resp = await client.post(
        "/api/v1/recommendations/events",
        json={"events": []},
    )
    assert resp.status_code in (401, 403, 422)


@pytest.mark.asyncio
async def test_events_batch_empty_rejected(
    client: AsyncClient, auth_headers: dict
):
    resp = await client.post(
        "/api/v1/recommendations/events",
        json={"events": []},
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_events_unknown_metadata_key_rejected(
    client: AsyncClient, auth_headers: dict, product_id: str
):
    resp = await client.post(
        "/api/v1/recommendations/events",
        json={
            "events": [{
                "type": "view",
                "product_id": product_id,
                "metadata": {"email": "x@x.com"},
            }]
        },
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_events_unknown_product_reported(
    client: AsyncClient, auth_headers: dict
):
    resp = await client.post(
        "/api/v1/recommendations/events",
        json={
            "events": [{
                "type": "view",
                "product_id": str(uuid.uuid4()),
                "ts": datetime.now(timezone.utc).isoformat(),
            }]
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 0
    assert body["rejected"] == 1
    assert body["errors"][0]["reason"] == "product_not_found"


@pytest.mark.asyncio
async def test_events_happy_path(
    client: AsyncClient,
    auth_headers: dict,
    product_id: str,
):
    resp = await client.post(
        "/api/v1/recommendations/events",
        json={
            "session_id": "sess-test",
            "events": [
                {"type": "view", "product_id": product_id},
                {
                    "type": "dwell",
                    "product_id": product_id,
                    "metadata": {"seconds": 42},
                },
                {
                    "type": "add_to_cart",
                    "product_id": product_id,
                    "metadata": {"qty": 1},
                },
            ],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] >= 1
    assert body["errors"] == []