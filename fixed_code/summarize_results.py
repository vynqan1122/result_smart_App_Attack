# -*- coding: utf-8 -*-
import csv
import glob
import json
import os

import matplotlib.pyplot as plt

def main():
    rows = []
    for path in glob.glob("results_json/**/*_results.json", recursive=True):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if "mean_success_rate" not in data:
            continue
        rows.append({
            "dataset": data.get("dataset", ""),
            "backbone": data.get("backbone", ""),
            "baseline_data": data.get("baseline_data", ""),
            "baseline": data.get("baseline", data.get("attack_type", "")),
            "attack_algo": data.get("attack_algo", data.get("attack_name", "")),
            "model_name": data.get("model_name", ""),
            "epsilon": data.get("epsilon", ""),
            "rounds": data.get("rounds", ""),
            "num_images": data.get("num_images", ""),
            "mean_success_rate": data.get("mean_success_rate", ""),
            "min_success_rate": data.get("min_success_rate", ""),
            "max_success_rate": data.get("max_success_rate", ""),
            "json_path": path,
        })

    os.makedirs("appendix_outputs", exist_ok=True)
    csv_path = "appendix_outputs/attack_summary.csv"
    fields = [
        "dataset", "backbone", "baseline_data", "baseline", "attack_algo",
        "model_name", "epsilon", "rounds", "num_images",
        "mean_success_rate", "min_success_rate", "max_success_rate", "json_path"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if rows:
        labels = ["{}-{}-{}-{}".format(r["dataset"], r["backbone"], r["baseline"], r["attack_algo"]) for r in rows]
        values = [float(r["mean_success_rate"]) for r in rows]
        plt.figure(figsize=(max(8, len(rows) * 0.6), 5))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
        plt.ylabel("Attack Success Rate")
        plt.tight_layout()
        plt.savefig("appendix_outputs/attack_success_summary.png", dpi=300)
        plt.close()

    print("Rows:", len(rows))
    print("Saved:", csv_path)
    print("Saved: appendix_outputs/attack_success_summary.png")

if __name__ == "__main__":
    main()
