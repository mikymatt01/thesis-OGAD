#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from community_graphs.sigl_loader import load_community_graphs_for_sigl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CommunityGraphs CSR pickle to SIGL dense ndarray pickle.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64", "uint8", "int8", "bool"))
    parser.add_argument("--max-dense-mib", type=float, default=None)
    parser.add_argument("--protocol", type=int, default=pickle.HIGHEST_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    max_bytes = None if args.max_dense_mib is None else int(args.max_dense_mib * 1024 * 1024)
    graphs = load_community_graphs_for_sigl(
        args.input,
        dtype=np.dtype(args.dtype),
        max_dense_bytes_per_graph=max_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(graphs, file, protocol=args.protocol)
    with args.output.open("rb") as file:
        reloaded = pickle.load(file)
    if not isinstance(reloaded, list) or not all(isinstance(graph, np.ndarray) for graph in reloaded):
        raise TypeError("Round-trip output did not contain list[np.ndarray]")
    shapes = sorted({graph.shape for graph in reloaded})
    dtypes = sorted({str(graph.dtype) for graph in reloaded})
    total_bytes = sum(int(graph.nbytes) for graph in reloaded)
    print("SIGL dense pickle generated")
    print(f"Graphs: {len(reloaded)}")
    print("Shapes: " + ", ".join(f"{r}x{c}" for r, c in shapes))
    print("Dtypes: " + ", ".join(dtypes))
    print(f"Total dense memory: {total_bytes / (1024 ** 2):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
