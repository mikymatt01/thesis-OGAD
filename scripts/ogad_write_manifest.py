#!/usr/bin/env python3
"""Write runs/<run_id>/manifest.json: experiment copy."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment_json", type=Path, help="Path to experiment JSON")
    p.add_argument("run_dir", type=Path, help="Run directory (must exist)")
    p.add_argument("run_id", help="Run identifier string")
    args = p.parse_args()

    experiment = json.loads(args.experiment_json.read_text(encoding="utf-8"))
    manifest: dict = {
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "experiment": experiment,
        "experiment_path": str(args.experiment_json.resolve()),
    }

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
