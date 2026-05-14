# -*- coding: utf-8 -*-
import json, pathlib, csv, os
import matplotlib.pyplot as plt

def main():
    rows = []
    for p in pathlib.Path("results_json").glob("**/*_results.json"):
        try: data = json.loads(p.read_text())
        except Exception: continue
        rows.append({"dataset": data.get("dataset"), "backbone": data.get("backbone"), "training_mode": data.get("training_mode"), "baseline": data.get("baseline"), "attack_algo": data.get("attack_algo"), "model_name": data.get("model_name"), "epsilon": data.get("epsilon"), "num_images": data.get("num_images"), "mean_success_rate": data.get("mean_success_rate"), "min_success_rate": data.get("min_success_rate"), "max_success_rate": data.get("max_success_rate"), "path": str(p)})
    os.makedirs("appendix_outputs", exist_ok=True); out_csv = "appendix_outputs/attack_summary.csv"
    fields = ["dataset","backbone","training_mode","baseline","attack_algo","model_name","epsilon","num_images","mean_success_rate","min_success_rate","max_success_rate","path"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("Saved:", out_csv, "rows:", len(rows))
    if rows:
        labels = [f"{r['dataset']}\n{r['backbone']}\n{r['training_mode']}\n{r['baseline']}\n{r['attack_algo']}" for r in rows]
        vals = [float(r["mean_success_rate"] or 0) for r in rows]
        plt.figure(figsize=(max(10, len(rows)*0.45), 6)); plt.bar(range(len(vals)), vals); plt.xticks(range(len(vals)), labels, rotation=90, fontsize=7); plt.ylabel("Attack Success Rate"); plt.tight_layout()
        out_png = "appendix_outputs/attack_success_summary.png"; plt.savefig(out_png, dpi=200); plt.close(); print("Saved:", out_png)
if __name__ == "__main__": main()
