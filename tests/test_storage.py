"""L1 — storage abstraction contract (LocalFS backend)."""

from __future__ import annotations

from pipeline.storage import LocalFS, get_storage


def test_roundtrip_bytes(tmp_path):
    s = LocalFS(tmp_path)
    s.write("a/b/c.bin", b"\x00\x01\x02")
    assert s.read("a/b/c.bin") == b"\x00\x01\x02"


def test_roundtrip_json(tmp_path):
    s = LocalFS(tmp_path)
    s.write_json("d/e.json", {"k": [1, 2, 3]})
    assert s.read_json("d/e.json") == {"k": [1, 2, 3]}


def test_exists(tmp_path):
    s = LocalFS(tmp_path)
    assert not s.exists("missing")
    s.write("here", b"x")
    assert s.exists("here")


def test_list_sorted_and_scoped(tmp_path):
    s = LocalFS(tmp_path)
    s.write("p/2", b"")
    s.write("p/1", b"")
    s.write("q/3", b"")
    assert s.list("p") == ["p/1", "p/2"]


def test_atomic_write_leaves_no_tmp(tmp_path):
    s = LocalFS(tmp_path)
    s.write("x/y.txt", b"hello")
    # No leftover temp files in the directory.
    leftovers = [p.name for p in (tmp_path / "x").iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_overwrite(tmp_path):
    s = LocalFS(tmp_path)
    s.write("k", b"one")
    s.write("k", b"two")
    assert s.read("k") == b"two"


def test_get_storage_localfs(tmp_path):
    assert isinstance(get_storage(str(tmp_path)), LocalFS)
    assert isinstance(get_storage(f"file://{tmp_path}"), LocalFS)
