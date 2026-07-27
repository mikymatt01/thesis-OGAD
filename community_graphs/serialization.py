from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scipy import sparse

from .hashing import file_sha256


VALIDATOR_SCRIPT = """#!/usr/bin/env python3
import sys
from pathlib import Path

for repo_candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (repo_candidate / "community_graphs").is_dir():
        if str(repo_candidate) not in sys.path:
            sys.path.insert(0, str(repo_candidate))
        break

from community_graphs.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
"""


def atomic_pickle_dump(graphs: list[sparse.csr_matrix], path: Path, protocol: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(graphs, file, protocol=protocol)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_text_atomic(text: str, path: Path, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(text)
        os.replace(tmp_name, path)
        if mode is not None:
            path.chmod(mode)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_validator_script(path: Path) -> None:
    write_text_atomic(VALIDATOR_SCRIPT, path, mode=0o755)


def copy_validator_script(path: Path) -> None:
    write_validator_script(path)


def sha256_for_written_pickles(output_dir: Path) -> dict[str, str]:
    return {
        "CommunityGraphs_train.pkl": file_sha256(output_dir / "CommunityGraphs_train.pkl"),
        "CommunityGraphs_test.pkl": file_sha256(output_dir / "CommunityGraphs_test.pkl"),
    }
