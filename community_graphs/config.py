from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DTYPE_CHOICES = ("uint8", "int8", "float32", "float64", "bool")
SIZE_MODE_CHOICES = ("fixed", "uniform", "linspace")


def parse_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_params(items: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid --params entry {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("Parameter keys must not be empty")
        params[key] = parse_value(value)
    return params


@dataclass
class DatasetConfig:
    family: str | None = None
    instances: int | None = None
    nodes: int | None = None
    nodes_min: int | None = None
    nodes_max: int | None = None
    size_mode: str = "fixed"
    params: dict[str, Any] = field(default_factory=dict)
    train_ratio: float | None = None
    train_instances: int | None = None
    test_instances: int | None = None
    shuffle_before_split: bool = False
    seed: int = 0
    output_dir: Path = Path("datasets/community_graphs")
    dataset_name: str | None = None
    dtype: str = "uint8"
    require_connected: bool = False
    largest_component: bool = False
    max_attempts: int = 100
    remove_self_loops: bool = True
    allow_self_loops: bool = False
    symmetrize: bool = True
    binary_adjacency: bool = True
    overwrite: bool = False
    workers: int = 1
    compute_metrics: bool = False
    protocol: int = pickle.HIGHEST_PROTOCOL
    dry_run: bool = False
    verbose: bool = False

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def load_config_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def merge_cli_with_file(cli: argparse.Namespace) -> DatasetConfig:
    file_data = load_config_file(cli.config) if cli.config else {}
    params = dict(file_data.get("params") or {})
    params.update(parse_params(cli.params))

    values: dict[str, Any] = {}
    for field_name in DatasetConfig.__dataclass_fields__:
        cli_name = field_name
        if hasattr(cli, cli_name):
            val = getattr(cli, cli_name)
            default = getattr(build_parser_defaults(), cli_name, None)
            if val != default and val is not None:
                values[field_name] = val
        if field_name in file_data and field_name not in values:
            values[field_name] = file_data[field_name]

    values["params"] = params
    if "output_dir" in values:
        values["output_dir"] = Path(values["output_dir"])
    return DatasetConfig(**values)


def build_parser_defaults() -> argparse.Namespace:
    return argparse.Namespace(
        family=None,
        instances=None,
        nodes=None,
        nodes_min=None,
        nodes_max=None,
        size_mode="fixed",
        params=None,
        train_ratio=None,
        train_instances=None,
        test_instances=None,
        shuffle_before_split=False,
        seed=0,
        output_dir=Path("datasets/community_graphs"),
        dataset_name=None,
        dtype="uint8",
        require_connected=False,
        largest_component=False,
        max_attempts=100,
        remove_self_loops=True,
        allow_self_loops=False,
        symmetrize=True,
        binary_adjacency=True,
        overwrite=False,
        workers=1,
        compute_metrics=False,
        protocol=pickle.HIGHEST_PROTOCOL,
        dry_run=False,
        verbose=False,
        config=None,
    )


def validate_protocol(protocol: int) -> None:
    if protocol < 0 or protocol > pickle.HIGHEST_PROTOCOL:
        raise ValueError(
            f"Pickle protocol {protocol} is not supported; valid range is 0..{pickle.HIGHEST_PROTOCOL}"
        )
