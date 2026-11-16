#!/usr/bin/env python3
"""Fetch the PNG version of Ludicon's 2026 Sponza cleanup, pinned and verified."""
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "c1da1669da43660b600fe352f3ccfdca6c89cb07"
SHA256 = "258b82df37b703cf383dfc58a64a547814f1b0445cfd9a89bd8867c176921eaf"
TARGET = Path(__file__).resolve().parents[1] / "assets/sponza/sponza-png.glb"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if TARGET.exists() and digest(TARGET) == SHA256:
    print(f"Already verified: {TARGET}")
else:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".part")
    try:
        urllib.request.urlretrieve(
            f"https://media.githubusercontent.com/media/ludicon/sponza-gltf/{COMMIT}/sponza-png.glb",
            temporary,
        )
        if digest(temporary) != SHA256:
            raise RuntimeError("Sponza checksum mismatch; download was not installed")
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {TARGET}")
