from __future__ import annotations
import os, csv
from typing import Dict, List
from ..utils.expressions import mask_expression

def _ensure_dir(p: str) -> None:
    """Ensure directory exists."""
    os.makedirs(p, exist_ok=True)

def _get_target_name(res: Dict) -> str:
    """Get target name from result dict (supports both 'target' and legacy 'target')."""
    return res.get("target", res.get("target", "unknown"))

def write_results_csvs(outdir: str, summary: Dict, full_group_columns: Dict[str, List[str]]) -> Dict[str, str]:
    """Write results to CSV files."""
    _ensure_dir(outdir)
    written = {}
    for res in summary.get("results", []):
        target = _get_target_name(res)
        # SF CSV
        sf_path = os.path.join(outdir, f"sf_topk_{target}.csv")
        with open(sf_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["target","group","mask_indices","expression","mean_loss","mean_rmse","mean_sign_acc","folds","per_fold_losses","per_fold_rmses","per_fold_signacc"])
            for g, rows in res["single_factor"].items():
                all_cols = full_group_columns.get(g, [])
                for row in rows:
                    expr = mask_expression(all_cols, row["columns"])
                    per = row.get("per_fold", {})
                    w.writerow([
                        target, g, "|".join(map(str, row["mask_indices"])), expr,
                        row["mean_loss"], row["mean_rmse"], row["mean_sign_acc"], row["folds"],
                        "|".join(map(str, per.get("losses", []))), "|".join(map(str, per.get("rmses", []))), "|".join(map(str, per.get("signacc", [])))
                    ])
        written[f"{target}_sf"] = sf_path

        # MF CSV
        mf_path = os.path.join(outdir, f"mf_leaderboard_{target}.csv")
        with open(mf_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["target","groups","expressions","mean_loss","mean_rmse","mean_sign_acc","folds","per_fold_losses","per_fold_rmses","per_fold_signacc"])
            for row in res.get("leaderboard", []):
                exprs = []
                for g, cols in zip(row["groups"], row["columns"]):
                    exprs.append(mask_expression(full_group_columns.get(g, []), cols))
                per = row.get("per_fold", {})
                w.writerow([
                    target, "|".join(row["groups"]), " || ".join(exprs),
                    row["mean_loss"], row["mean_rmse"], row["mean_sign_acc"], row["folds"],
                    "|".join(map(str, per.get("losses", []))), "|".join(map(str, per.get("rmses", []))), "|".join(map(str, per.get("signacc", [])))
                ])
        written[f"{target}_mf"] = mf_path

        # Holdout CSV
        import numpy as np
        h = res.get("holdout", {}); mh = h.get("metrics", {})
        ho_path = os.path.join(outdir, f"holdout_{target}.csv")
        with open(ho_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            realized = h.get("realized", [])
            predicted = h.get("predicted", [])
            simulated = h.get("simulated", [])
            # Compute cumulative versions
            realized_cum = np.cumsum(realized).tolist() if realized else []
            predicted_cum = np.cumsum(predicted).tolist() if predicted else []
            w.writerow(["idx","realized","predicted","simulated","realized_cum","predicted_cum"])
            for idx, r, p, s, rc, pc in zip(h.get("index", []), realized, predicted, simulated, realized_cum, predicted_cum):
                w.writerow([idx, r, p, s, rc, pc])
            w.writerow([]); w.writerow(["metric","value"])
            for k, v in mh.items():
                w.writerow([k, v])
        written[f"{target}_holdout"] = ho_path
    return written

