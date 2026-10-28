from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import MODEL_CHOICES, generate_many_baselines, load_graphs, save_graphs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit and generate classical graph baselines.")
    parser.add_argument("--config", type=Path, help="Normalized experiment JSON.")
    parser.add_argument("--model", choices=("all", *MODEL_CHOICES), default=None)
    parser.add_argument("--train-graphs", type=Path)
    parser.add_argument("--test-graphs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-graphs", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def _resolve_from_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        missing = [
            name
            for name in ("train_graphs", "test_graphs", "output_dir")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "Missing required argument(s) without --config: {}".format(
                    ", ".join("--" + name.replace("_", "-") for name in missing)
                )
            )
        args.model = args.model or "all"
        args.num_graphs = args.num_graphs or 100
        args.seed = args.seed or 0
        return args

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = config.get("dataset") or {}
    ggme = config.get("ggme") or {}
    baselines = config.get("baselines") or {}
    args.train_graphs = args.train_graphs or Path(dataset.get("train_graphs_path") or config.get("sigl", {}).get("Adjs_dir"))
    args.test_graphs = args.test_graphs or Path(dataset.get("test_graphs_path") or ggme.get("test_graphs"))
    args.output_dir = args.output_dir or Path(
        baselines.get("output_dir", "/app/data/predictions/graphs/classical")
    )
    args.num_graphs = args.num_graphs or int(
        ggme.get("num_generated_graphs") or dataset.get("generated_graph_count") or 100
    )
    args.seed = args.seed if args.seed is not None else int(baselines.get("seed", ggme.get("seed", 0)))
    if args.model is None:
        models = baselines.get("models") or list(MODEL_CHOICES)
        args.model = "all" if list(models) == list(MODEL_CHOICES) else list(models)
    return args


def _model_names(model_arg) -> tuple[str, ...]:
    if model_arg == "all":
        return MODEL_CHOICES
    if isinstance(model_arg, str):
        return (model_arg,)
    names = tuple(str(model) for model in model_arg)
    unknown = [name for name in names if name not in MODEL_CHOICES]
    if unknown:
        raise ValueError("Unknown baseline model(s): {}".format(", ".join(unknown)))
    return names


def main(argv: list[str] | None = None) -> int:
    args = _resolve_from_config(build_parser().parse_args(argv))
    train_graphs = load_graphs(args.train_graphs)
    test_graphs = load_graphs(args.test_graphs)
    model_names = _model_names(args.model)
    generated = generate_many_baselines(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        model_names=model_names,
        num_graphs=args.num_graphs,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model_name, graphs in generated.items():
        path = args.output_dir / f"{model_name}.pickle"
        save_graphs(graphs, path)
        outputs[model_name] = str(path)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0
