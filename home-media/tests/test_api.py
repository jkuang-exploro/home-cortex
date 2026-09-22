from __future__ import annotations

from conftest import make_jpeg, make_video


def test_health_refresh_list_filters_pagination_and_lookup(client, settings):
    make_jpeg(settings.photo_root / "old.jpg", captured_at="2020:01:01 00:00:00")
    make_jpeg(settings.photo_root / "new.jpg", captured_at="2024:01:01 00:00:00")
    make_video(settings.video_root / "clip.mp4")

    assert client.get("/health").json()["status"] == "ok"
    refresh = client.post("/refresh")
    assert refresh.status_code == 200
    assert refresh.json()["added"] == 3

    first = client.get("/items", params={"type": "all", "limit": 2}).json()
    assert first["total"] == 3
    assert len(first["items"]) == 2
    assert first["next_offset"] == 2
    assert first["items"][0]["filename"] == "clip.mp4"
    assert not any("/media/" in str(value) for value in first["items"][0].values())

    photos = client.get("/items", params={"type": "photo", "limit": 10}).json()
    assert [item["filename"] for item in photos["items"]] == ["new.jpg", "old.jpg"]
    assert all(item["type"] == "photo" for item in photos["items"])
    item = client.get(f"/items/{photos['items'][0]['id']}")
    assert item.status_code == 200
    assert item.json()["content_url"].startswith("/media-api/items/media_")

    videos = client.get("/items", params={"type": "video"}).json()
    assert videos["total"] == 1
    assert videos["items"][0]["duration_seconds"] is not None


def test_unknown_and_path_shaped_ids_are_rejected(client):
    for path in (
        "/items/media_000000000000000000000000/content",
        "/items/..%2F..%2Fetc%2Fpasswd/content",
        "/items/%2Fetc%2Fpasswd/content",
    ):
        assert client.get(path).status_code in {404, 422}


def test_original_delivery_and_byte_ranges(client, settings):
    path = settings.video_root / "range.mp4"
    make_video(path)
    client.post("/refresh")
    item = client.get("/items", params={"type": "video"}).json()["items"][0]
    url = item["content_url"].removeprefix("/media-api")

    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    partial = client.get(url, headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206
    assert partial.content == path.read_bytes()[:10]
    assert partial.headers["content-range"] == f"bytes 0-9/{path.stat().st_size}"
    invalid = client.get(url, headers={"Range": "bytes=999999-1000000"})
    assert invalid.status_code == 416


def test_corrupt_thumbnail_has_controlled_failure(client, settings):
    (settings.photo_root / "bad.jpg").write_bytes(b"broken")
    client.post("/refresh")
    item = client.get("/items").json()["items"][0]
    response = client.get(item["thumbnail_url"].removeprefix("/media-api"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "thumbnail_failed"

