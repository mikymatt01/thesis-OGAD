from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DTYPE_CHOICES, SIZE_MODE_CHOICES, DatasetConfig, merge_cli_with_file
from .generation import generate_dataset
from .registry import canonical_family_slugs, get_family


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic CommunityGraphs datasets.")
    parser.add_argument("--family")
    parser.add_argument("--instances", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--nodes-min", dest="nodes_min", type=int)
    parser.add_argument("--nodes-max", dest="nodes_max", type=int)
    parser.add_argument("--size-mode", choices=SIZE_MODE_CHOICES, default="fixed")
    parser.add_argument("--params", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument("--train-ratio", dest="train_ratio", type=float)
    parser.add_argument("--train-instances", dest="train_instances", type=int)
    parser.add_argument("--test-instances", dest="test_instances", type=int)
    parser.add_argument("--shuffle-before-split", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=Path("datasets/community_graphs"))
    parser.add_argument("--dataset-name", dest="dataset_name")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="uint8")
    parser.add_argument("--require-connected", action="store_true")
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--remove-self-loops", dest="remove_self_loops", action="store_true", default=True)
    parser.add_argument("--allow-self-loops", action="store_true")
    parser.add_argument("--symmetrize", action="store_true", default=False)
    parser.add_argument("--binary-adjacency", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--protocol", type=int, default=-1)
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--describe-family")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.protocol == -1:
        import pickle

        args.protocol = pickle.HIGHEST_PROTOCOL
    if args.list_families:
        for slug in canonical_family_slugs():
            print(slug)
        return 0
    if args.describe_family:
        print(get_family(args.describe_family).describe())
        return 0
    try:
        config = merge_cli_with_file(args)
        result = generate_dataset(config, command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])])
    except Exception as exc:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logging.error("%s", exc)
        return 1
    if config.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        splits = result["splits"]
        print("CommunityGraphs dataset generated")
        print(f"Output: {config.output_dir}")
        print(f"Train graphs: {splits['train_instances']}")
        print(f"Test graphs: {splits['test_instances']}")
        print("Matrix type: scipy.sparse.csr_matrix")
    return 0
