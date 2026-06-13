"""Shared fixtures: a real (minimal) PDF and a LocalFS-backed job context.

The fixture PDF is generated in pure Python (correct xref offsets) so the repo
carries no binary and the tests run identically on any machine.
"""

from __future__ import annotations

import pytest

from pipeline.jobctx import JobContext
from pipeline.run import init_job
from pipeline.storage import LocalFS
from pipeline.vlm import VLMClient


def make_minimal_pdf() -> bytes:
    """Build a valid one-page PDF with a line of text, computing xref offsets."""
    stream = b"BT /F1 24 Tf 72 700 Td (Hello Cryogenic) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_pos)
    return bytes(out)


def make_blank_pdf() -> bytes:
    """A valid one-page PDF with NO text layer — exercises the scanned branch."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_pos)
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_minimal_pdf())
    return path


@pytest.fixture
def blank_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_blank_pdf())
    return path


@pytest.fixture
def storage(tmp_path):
    return LocalFS(tmp_path / "store")


@pytest.fixture
def mock_vlm():
    return VLMClient(base_url="mock://", model="mock")


@pytest.fixture
def job(storage, sample_pdf) -> JobContext:
    """An initialised job (source copied, manifest written), before stages run."""
    return init_job(storage, sample_pdf)
