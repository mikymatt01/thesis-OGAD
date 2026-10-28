from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse

from .csr import normalize_csr


def csr_structure_sha256(adjacency: sparse.csr_matrix) -> str:
    csr = normalize_csr(adjacency.copy())
    digest = hashlib.sha256()
    digest.update(str(tuple(csr.shape)).encode("utf-8"))
    digest.update(np.dtype(csr.dtype).str.encode("utf-8"))
    digest.update(csr.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(csr.indices.astype(np.int64, copy=False).tobytes())
    digest.update(np.ascontiguousarray(csr.data).tobytes())
    return digest.hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
