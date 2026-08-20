"""Unit tests for transcript capture (share-bundle export) + central return
(artifact-store upload / publish_bundle). No network — httpx.MockTransport.
"""

from __future__ import annotations

import tarfile

import httpx
import pytest
from silico_eval import LabClient, LabClientError, default_publish_dest, publish_bundle
from silico_eval import lab_client as lc


def _client(handler) -> LabClient:
    c = LabClient("https://pod.example", "tok")
    c._client = httpx.Client(
        base_url=c.pod_url,
        headers={"Authorization": "Bearer tok"},
        transport=httpx.MockTransport(handler),
    )
    return c


def _roots_client(roots):
    return _client(lambda r: httpx.Response(200, json={"roots": roots}))


def test_default_publish_dest_prefers_store_then_shared():
    c = _roots_client(["/tmp", "/srv/silico-state/_shared", "/artifact-store/shared"])
    assert default_publish_dest(c) == "/artifact-store/shared/eval-bundles"


def test_default_publish_dest_falls_back_to_shared_tree():
    # No cloud store root (Q4 not provisioned) -> the per-tenant _shared tree.
    c = _roots_client(
        ["/srv/silico-state/users/u-x/artifacts", "/srv/silico-state/_shared", "/tmp"]
    )
    assert default_publish_dest(c) == "/srv/silico-state/_shared/eval-bundles"


def test_default_publish_dest_avoids_tmp_when_possible():
    c = _roots_client(["/tmp", "/srv/silico-state/users/u-x/artifacts"])
    assert default_publish_dest(c) == "/srv/silico-state/users/u-x/artifacts/eval-bundles"


def test_default_publish_dest_raises_with_no_roots():
    c = _roots_client([])
    with pytest.raises(LabClientError):
        default_publish_dest(c)


def test_export_share_bundle_returns_raw_bytes():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"PK\x03\x04sil-zip-bytes")

    c = _client(handler)
    data = c.export_share_bundle("default", "exp_1")
    assert data == b"PK\x03\x04sil-zip-bytes"
    assert captured["url"].endswith("/api/workspaces/default/experiments/exp_1/share")


def test_upload_artifact_posts_multipart():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = request.content.decode("latin-1")
        captured["has_dest"] = "destination_path" in body and "/store/evals" in body
        captured["has_file"] = "hello.txt" in body and "data-bytes" in body
        return httpx.Response(200, json={"ok": True})

    c = _client(handler)
    out = c.upload_artifact("/store/evals", {"hello.txt": b"data-bytes"})
    assert out == {"ok": True}
    assert captured["url"].endswith("/api/artifacts/upload")
    assert captured["has_dest"] and captured["has_file"]


def test_upload_artifact_raises_on_error():
    c = _client(lambda r: httpx.Response(429, json={"error": "too many"}))
    with pytest.raises(LabClientError) as exc:
        c.upload_artifact("/store/evals", {"x": b"y"})
    assert exc.value.status == 429


def test_store_upload_full_flow(monkeypatch):
    # Mocks the Lab control endpoints (create/sign/record) and the presigned
    # direct-to-S3 PUT, verifying store_upload drives the whole sequence.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/api/artifact-store/uploads":  # create
            return httpx.Response(
                200,
                json={
                    "id": "sess1",
                    "files": [
                        {"client_id": "0", "size": 3, "part_size": 5_000_000, "part_count": 1}
                    ],
                },
            )
        if path.endswith("/parts/sign"):
            return httpx.Response(
                200,
                json={
                    "parts": [
                        {
                            "client_id": "0",
                            "part_number": 1,
                            "url": "https://s3.fake/put",
                            "expires_at": "z",
                        }
                    ]
                },
            )
        if path.endswith("/parts/record"):
            return httpx.Response(
                200,
                json={
                    "state": "completed",
                    "object_refs": ["artifact://t/eval-bundles/e/b.tar.gz"],
                },
            )
        return httpx.Response(404, json={})

    put_seen = {}

    def fake_put(url, content=None, timeout=None):
        put_seen["url"] = url
        put_seen["body"] = content
        return httpx.Response(200, headers={"ETag": '"abc123"'})

    c = _client(handler)
    monkeypatch.setattr(lc.httpx, "put", fake_put)
    result = c.store_upload("eval-bundles/e", {"b.tar.gz": b"\x00\x01\x02"})

    assert result["state"] == "completed"
    assert put_seen["url"] == "https://s3.fake/put" and put_seen["body"] == b"\x00\x01\x02"
    assert "/api/artifact-store/uploads" in calls
    assert any(p.endswith("/parts/sign") for p in calls)
    assert any(p.endswith("/parts/record") for p in calls)


def test_store_ref_to_path_drops_tenant():
    assert (
        LabClient.store_ref_to_path("artifact://tenantX/eval-bundles/exp_1/b.tar.gz")
        == "/artifact-store/eval-bundles/exp_1/b.tar.gz"
    )


def test_publish_bundle_targets_store_and_returns_pull_path(tmp_path, monkeypatch):
    bdir = tmp_path / "bundle_exp1"
    (bdir / "files").mkdir(parents=True)
    (bdir / "manifest.json").write_text('{"eid": "exp_1"}')
    (bdir / "files" / "x.bin").write_bytes(b"\x00\x01\x02")

    captured = {}

    def fake_store_upload(destination, files):
        captured["destination"] = destination
        captured["files"] = files  # {name: tar_bytes}
        return {
            "state": "completed",
            "object_refs": [f"artifact://goodfire-evals/{destination}/bundle_exp1.tar.gz"],
        }

    c = _client(lambda r: httpx.Response(200, json={}))
    monkeypatch.setattr(c, "store_upload", fake_store_upload)

    res = publish_bundle(c, str(bdir), eid="exp_1", log=lambda *a: None)
    # Default destination is per-eid under eval-bundles/, and the pull path is
    # the /artifact-store/ pseudo path derived from the returned ref.
    assert captured["destination"] == "eval-bundles/exp_1"
    assert res["ok"] is True
    assert res["path"] == "/artifact-store/eval-bundles/exp_1/bundle_exp1.tar.gz"

    # The uploaded blob is a valid tar.gz of the bundle dir.
    (tar_name, tar_bytes) = next(iter(captured["files"].items()))
    out = tmp_path / tar_name
    out.write_bytes(tar_bytes)
    with tarfile.open(str(out), "r:gz") as tf:
        names = {m.name for m in tf.getmembers()}
    assert any(n.endswith("manifest.json") for n in names)
    assert any(n.endswith("files/x.bin") for n in names)


def test_publish_bundle_falls_back_to_efs_when_store_endpoint_absent(tmp_path, monkeypatch):
    # Deploy-lag: the store multipart endpoint 404s on an older image -> EFS fallback.
    bdir = tmp_path / "bundle_exp1"
    bdir.mkdir()
    (bdir / "manifest.json").write_text("{}")

    def store_upload_404(destination, files):
        raise LabClientError("not found", status=404)

    captured = {}

    def fake_upload_artifact(dest, files):
        captured["dest"] = dest
        captured["names"] = list(files)
        return {"ok": True}

    c = _client(lambda r: httpx.Response(200, json={"roots": ["/srv/silico-state/_shared"]}))
    monkeypatch.setattr(c, "store_upload", store_upload_404)
    monkeypatch.setattr(c, "upload_artifact", fake_upload_artifact)

    res = publish_bundle(c, str(bdir), eid="exp_1", log=lambda *a: None)
    assert res["via"] == "efs-fallback" and res["ok"] is True
    assert captured["dest"] == "/srv/silico-state/_shared/eval-bundles"
    assert res["path"] == "/srv/silico-state/_shared/eval-bundles/bundle_exp1.tar.gz"


def test_publish_bundle_reraises_non_deploy_lag_errors(tmp_path, monkeypatch):
    bdir = tmp_path / "b"
    bdir.mkdir()
    (bdir / "manifest.json").write_text("{}")

    def store_upload_403(destination, files):
        raise LabClientError("forbidden", status=403)

    c = _client(lambda r: httpx.Response(200, json={}))
    monkeypatch.setattr(c, "store_upload", store_upload_403)
    with pytest.raises(LabClientError):
        publish_bundle(c, str(bdir), eid="b", log=lambda *a: None)


def test_pull_bundle_downloads_and_extracts(tmp_path, monkeypatch):
    import io

    # Build a tar.gz that the fake download will "return".
    payload = tmp_path / "src"
    (payload).mkdir()
    (payload / "manifest.json").write_text("{}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(payload / "manifest.json"), arcname="bundle/manifest.json")
    tar_bytes = buf.getvalue()

    c = _client(lambda r: httpx.Response(200, json={}))

    def fake_download_to(abs_path, dest, *, fabric_id=None, max_bytes=None):
        assert abs_path == "/artifact-store/eval-bundles/exp_1/bundle_exp1.tar.gz"
        assert fabric_id == "artifact-store"
        dest.write(tar_bytes)
        return len(tar_bytes)

    monkeypatch.setattr(c, "download_to", fake_download_to)
    out = tmp_path / "pulled"
    from silico_eval import pull_bundle

    pull_bundle(c, "/artifact-store/eval-bundles/exp_1/bundle_exp1.tar.gz", str(out))
    assert (out / "bundle_exp1.tar.gz").exists()
    assert (out / "bundle" / "manifest.json").exists()  # extracted
