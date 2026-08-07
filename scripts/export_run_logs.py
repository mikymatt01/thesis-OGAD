#!/usr/bin/env python3
"""Esporta in runs_export/ i log e le metriche dei soli batch che usano i
dataset baseline generati da noi (scripts/build_validation_datasets.sh).

I batch che girano solo su CommunityGraphs sono esclusi.

Lo script e' in sola lettura su runs/: gli originali restano intatti sul disco
per analisi successive, in runs_export/ finisce solo la copia da versionare.

Uso:
    python scripts/export_run_logs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
EXPORT = REPO / "runs_export"

# I 5 dataset della suite di validazione (vedi datasets/README.md).
BASELINE_DATASETS = (
    "ba_n100_m3",
    "er_n100_p005",
    "expdeg_n100_lin",
    "sbm_n100_4c",
    "ws_n100_k8_p01",
)

# Log per-job dei worker SIGL (decine di migliaia di file da ~600 byte) e
# logging interno di wandb: rumore, restano solo in runs/.
EXCLUDED_DIRS = ("sigl_workers", "wandb")

TEXT_LOG_SUFFIXES = (".log",)
METRIC_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml")

# Righe di avanzamento tqdm: "45%|####      | 90/200 [00:56<01:08,  1.60it/s]".
TQDM_RE = re.compile(r"\d+%\|" r"|\|\s*\d+/\d+\s*\[")


def uses_baseline_datasets(batch: Path) -> set[str]:
    """Dataset baseline usati dal batch, dedotti dai path e dagli experiment json."""
    found = set()
    for path in batch.rglob("*"):
        name = path.name
        for dataset in BASELINE_DATASETS:
            if dataset in name:
                found.add(dataset)
        if found == set(BASELINE_DATASETS):
            break
    return found


def strip_progress_bars(text: str) -> tuple[str, int]:
    """Rimuove lo spam tqdm tenendo l'ultimo aggiornamento di ogni barra.

    Ogni barra si aggiorna via carriage return, quindi un log da 20 MB e' quasi
    tutto ridisegni della stessa riga. Tenendo solo l'ultimo frame di ogni
    sequenza si conservano totale iterazioni e tempi, che servono per i tempi
    di training, e si buttano via i frame intermedi.
    """
    lines = text.replace("\r", "\n").split("\n")
    out: list[str] = []
    burst: list[str] = []
    dropped = 0
    blank_run = 0

    for line in lines:
        if TQDM_RE.search(line):
            burst.append(line)
            continue
        if burst:
            out.append(burst[-1])
            dropped += len(burst) - 1
            burst = []
        if line.strip():
            blank_run = 0
            out.append(line)
        else:
            # Lo split su \r genera righe vuote spurie: ne tiene al massimo una.
            blank_run += 1
            if blank_run == 1:
                out.append("")

    if burst:
        out.append(burst[-1])
        dropped += len(burst) - 1

    return "\n".join(out).rstrip("\n") + "\n", dropped


def should_copy(path: Path) -> bool:
    parts = set(path.parts)
    if parts & set(EXCLUDED_DIRS):
        return False
    suffix = path.suffix.lower()
    if suffix in TEXT_LOG_SUFFIXES:
        return True
    if suffix in METRIC_SUFFIXES:
        # I bootstrap_metadata.json sotto data/train/graphons sono input
        # rigenerabili da decine di MB, non metriche di run.
        return "data" not in parts
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batches = []
    for batch in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        datasets = uses_baseline_datasets(batch)
        if datasets:
            batches.append((batch, sorted(datasets)))

    if not args.dry_run and EXPORT.exists():
        shutil.rmtree(EXPORT)

    total_src = total_dst = n_files = n_dropped = 0
    rows = []

    for batch, datasets in batches:
        b_src = b_dst = b_files = 0
        for path in sorted(batch.rglob("*")):
            if not path.is_file() or not should_copy(path):
                continue
            rel = path.relative_to(RUNS)
            dst = EXPORT / rel
            src_size = path.stat().st_size

            if path.suffix.lower() in TEXT_LOG_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                cleaned, dropped = strip_progress_bars(text)
                if dropped:
                    cleaned += (
                        f"\n[export] {dropped} frame intermedi di progress bar "
                        f"rimossi; log integrale in runs/{rel}\n"
                    )
                n_dropped += dropped
                payload = cleaned.encode("utf-8")
            else:
                payload = path.read_bytes()

            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(payload)

            b_src += src_size
            b_dst += len(payload)
            b_files += 1

        rows.append((batch.name, datasets, b_files, b_src, b_dst))
        total_src += b_src
        total_dst += b_dst
        n_files += b_files
        print(
            f"{batch.name:28s} {b_files:6d} file  "
            f"{b_src / 1048576:8.1f} MB -> {b_dst / 1048576:7.2f} MB  "
            f"[{', '.join(datasets)}]"
        )

    # Report aggregati prodotti da scripts/rank_runs (riguardano questi batch).
    reports = sorted(p for p in RUNS.glob("*") if p.is_file() and p.suffix in (".csv", ".md"))
    for report in reports:
        if not args.dry_run:
            EXPORT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report, EXPORT / report.name)
        total_dst += report.stat().st_size
        n_files += 1
    print(f"{'(report top-level)':28s} {len(reports):6d} file")

    if not args.dry_run:
        write_manifest(rows, reports)

    print(
        f"\nTOTALE: {n_files} file, {total_src / 1048576:.1f} MB in runs/ "
        f"-> {total_dst / 1048576:.2f} MB in runs_export/ "
        f"({n_dropped} frame di progress bar rimossi)"
    )
    return 0


def write_manifest(rows, reports) -> None:
    lines = [
        "# Log delle run esportati per il repo",
        "",
        "Sottoinsieme versionato di `runs/`, prodotto da `scripts/export_run_logs.py`.",
        "`runs/` resta locale e ignorato da git: qui ci sono solo i log e le metriche",
        "dei batch che girano sui **dataset baseline generati da noi**",
        "(`datasets/README.md`). I batch su soli CommunityGraphs sono esclusi.",
        "",
        "## Batch inclusi",
        "",
        "| batch | dataset | file | dimensione |",
        "|---|---|---:|---:|",
    ]
    for name, datasets, n_files, _src, dst in rows:
        lines.append(
            f"| `{name}` | {', '.join(f'`{d}`' for d in datasets)} | "
            f"{n_files} | {dst / 1048576:.2f} MB |"
        )
    lines += [
        "",
        "## Cosa e' incluso",
        "",
        "- `*.log` con i frame intermedi delle progress bar tqdm rimossi (l'ultimo",
        "  frame di ogni barra resta, per conservare iterazioni e tempi);",
        "- `*.json` / `*.jsonl` / `*.yaml` di metriche e configurazione",
        "  (`metrics.jsonl`, `manifest.json`, `experiment.json`, `status.json`, hydra);",
        "- i report aggregati CSV/MD al livello di `runs/`:",
        "  " + ", ".join(f"`{r.name}`" for r in reports) + ".",
        "",
        "## Cosa e' escluso (resta solo in `runs/`)",
        "",
        "- pickle, PNG, GIF, `.npz`, `generated_samples*.txt`: artefatti binari pesanti;",
        "- `sigl_workers/job_*.log`: decine di migliaia di file da ~600 byte;",
        "- directory `wandb/`: logging interno dell'SDK;",
        "- `data/train/graphons/bootstrap_metadata.json`: input rigenerabile da decine di MB.",
        "",
        "Per rigenerare questa directory: `python scripts/export_run_logs.py`.",
        "",
    ]
    (EXPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
