import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_products_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/products")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_product_not_found(client: AsyncClient) -> None:
    resp = await client.get("/api/products/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_batches_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/batches")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_batch_not_found(client: AsyncClient) -> None:
    resp = await client.get("/api/batches/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_analyze_rejects_invalid_mime(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/analyze",
        data={"product_name": "Test"},
        files={"image": ("test.pdf", b"%PDF fake", "application/pdf")},
    )
    assert resp.status_code == 415
