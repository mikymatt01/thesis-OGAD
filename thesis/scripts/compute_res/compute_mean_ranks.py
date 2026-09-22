#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

INPUT = Path("ogad_experiment_results.csv")
OUT_BY_METRIC = Path("mean_rank_by_metric.csv")
OUT_BY_DATASET = Path("mean_rank_by_dataset.csv")
METRICS = ["Degree_MMD", "Clustering_MMD", "NormLaplacian_MMD"]

def average_ranks(values_by_model):
    """Ascending ranks; tied values receive the average occupied rank."""
    ordered = sorted(values_by_model.items(), key=lambda x: x[1])
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks

def load_results(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "Dataset": row["Dataset"],
                "Model": row["Model"],
                **{m: float(row[m]) for m in METRICS},
            })
    return rows

def elementary_ranks(rows):
    values = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        for metric in METRICS:
            values[row["Dataset"]][metric][row["Model"]] = row[metric]

    ranks = defaultdict(dict)
    for dataset in values:
        for metric in METRICS:
            ranks[dataset][metric] = average_ranks(values[dataset][metric])
    return ranks

def main():
    rows = load_results(INPUT)
    ranks = elementary_ranks(rows)
    datasets = sorted(ranks.keys())
    models = sorted({r["Model"] for r in rows})

    # 1) Mean rank across datasets, separately for each metric.
    by_metric = {}
    for model in models:
        by_metric[model] = {
            metric: sum(ranks[d][metric][model] for d in datasets) / len(datasets)
            for metric in METRICS
        }

    # 2) Mean rank across the 3 metrics, separately for each dataset.
    by_dataset = {}
    for model in models:
        by_dataset[model] = {
            dataset: sum(ranks[dataset][metric][model] for metric in METRICS) / len(METRICS)
            for dataset in datasets
        }

    with OUT_BY_METRIC.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model","DD_Mean_Rank","CC_Mean_Rank","NormLaplacian_Mean_Rank"])
        for model in models:
            w.writerow([
                model,
                f"{by_metric[model]['Degree_MMD']:.4f}",
                f"{by_metric[model]['Clustering_MMD']:.4f}",
                f"{by_metric[model]['NormLaplacian_MMD']:.4f}",
            ])

    with OUT_BY_DATASET.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + datasets)
        for model in models:
            w.writerow([model] + [f"{by_dataset[model][d]:.4f}" for d in datasets])

    print("Mean rank by metric:")
    for model in models:
        print(model, {k: round(v, 2) for k, v in by_metric[model].items()})

    print("\nMean rank by dataset:")
    for model in models:
        print(model, {k: round(v, 2) for k, v in by_dataset[model].items()})

if __name__ == "__main__":
    main()
