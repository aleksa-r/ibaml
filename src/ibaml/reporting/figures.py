from __future__ import annotations
from typing import Dict, List, Optional
import os, json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime


def _ensure_dir(p: str) -> None:
    """Ensure directory exists."""
    os.makedirs(p, exist_ok=True)


def create_beeswarm_importance(feature_importances: Dict[str, float], output_path: str, title: str = "Feature Importance") -> Optional[str]:
    """Create feature importance visualization."""
    if not feature_importances:
        with open(output_path, "w") as f: f.write("")
        return output_path
    
    try:
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        if not features:
            with open(output_path, "w") as f: f.write("")
            return output_path
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(features))
        colors = ['#1f77b4' if imp > 0 else '#d62728' for imp in importances]
        ax.bar(x_pos, importances, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Importance (Gain)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linewidth=1)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error creating beeswarm: {e}")
        with open(output_path, "w") as f: f.write("")
        return output_path


def save_holdout_figure(figdir: str, target: str, holdout: Dict) -> str:
    """Create and save holdout performance figure."""
    _ensure_dir(figdir)
    path = os.path.join(figdir, f"{target}_holdout.png")
    import numpy as np
    # X-axis: prefer dates if provided
    idx = holdout.get("index", []) or []
    dates = []
    if idx:
        try:
            dates = [datetime.fromisoformat(str(d)) for d in idx]
        except Exception:
            dates = []
    xs = dates if dates else list(range(len(holdout.get("realized", []))))
    
    # Get monthly returns
    realized = holdout.get("realized", [])
    predicted = holdout.get("predicted", [])
    # 'simulated' provided by pipeline is already a wealth index series
    simulated = holdout.get("simulated", [])
    
    # Compute wealth-indexed cumulative versions for plotting
    realized_cum = (np.cumprod(1.0 + np.array(realized, dtype=float)) - 1.0).tolist() if realized else []
    predicted_cum = (np.cumprod(1.0 + np.array(predicted, dtype=float)) - 1.0).tolist() if predicted else []
    
    plt.figure(figsize=(12, 6))
    plt.plot(xs, realized_cum, label="realized (cumulative)", marker='o', linewidth=2)
    plt.plot(xs, predicted_cum, label="predicted (cumulative)", marker='s', linewidth=2)
    plt.plot(xs, simulated, label="simulated (signal-gated cumulative)", marker='^', linewidth=2)
    plt.title(f"{target} – Holdout Performance")
    plt.ylabel("Cumulative Return")
    if dates:
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Date")
    else:
        plt.xlabel("Month")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    return path


def save_cv_bar(figdir: str, target: str, what: str, labels: List[str], losses: List[float]) -> str:
    """Create and save cross-validation fold loss bar chart."""
    _ensure_dir(figdir)
    path = os.path.join(figdir, f"{target}_{what}_cv.png")
    plt.figure()
    plt.bar(labels, losses)
    plt.title(f"{target} – {what} CV fold losses"); plt.xlabel("fold"); plt.ylabel("loss"); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    return path


def save_fi_bar(figdir: str, target: str, fi_map: Dict[str, float]) -> str:
    """Create feature importance bar chart.
    
    Handles both individual feature importances and group-level aggregates.
    """
    _ensure_dir(figdir)
    path = os.path.join(figdir, f"{target}_feature_importance.png")
    if not fi_map:
        # Create a placeholder PNG instead of an empty file (prevents PNG corruption)
        plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, "No feature importance available", ha='center', va='center')
        plt.axis('off')
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path
    
    # Separate individual features from group aggregates
    individual_items = [(k, v) for k, v in fi_map.items() if not k.endswith('_gbp')]
    group_items = [(k, v) for k, v in fi_map.items() if k.endswith('_gbp')]
    
    # Use individual features if available, otherwise use group aggregates
    items_to_plot = individual_items if individual_items else group_items
    
    if not items_to_plot:
        plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, "No feature importance available", ha='center', va='center')
        plt.axis('off')
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
        return path
    
    items = sorted(items_to_plot, key=lambda kv: kv[1], reverse=True)[:20]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(8, max(3, 0.35*len(labels))))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title(f"{target} – Feature importance (gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    return path


def save_fi_beeswarm(figdir: str, target: str, fi_map: Dict[str, float],
                     shap_values: Optional[np.ndarray] = None,
                     shap_X: Optional[np.ndarray] = None,
                     shap_feature_names: Optional[List[str]] = None) -> str:
    """Create SHAP beeswarm plot showing feature importance.
    
    If SHAP values are provided, creates a proper SHAP beeswarm plot.
    Otherwise falls back to a bar chart of gain-based feature importance.
    
    Args:
        figdir: Output directory for figures
        target: target name
        fi_map: Feature importance dictionary (gain-based, fallback)
        shap_values: SHAP values array of shape (n_samples, n_features)
        shap_X: Feature values array of shape (n_samples, n_features)
        shap_feature_names: List of feature names
    
    Returns:
        Path to saved figure
    """
    _ensure_dir(figdir)
    path = os.path.join(figdir, f"{target}_fi_beeswarm.png")
    
    # Try to use SHAP beeswarm if values are available
    if shap_values is not None and shap_X is not None and shap_feature_names is not None:
        try:
            import shap
            shap_values = np.array(shap_values)
            shap_X = np.array(shap_X)
            
            # Create SHAP Explanation object
            explanation = shap.Explanation(
                values=shap_values,
                data=shap_X,
                feature_names=shap_feature_names
            )
            
            # Create beeswarm plot
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(explanation, show=False, max_display=15)
            plt.title(f"{target} – SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
        except ImportError:
            pass  # Fall back to gain-based plot
        except Exception as e:
            print(f"SHAP beeswarm failed for {target}: {e}")
            # Fall back to gain-based plot
    
    # Fallback: create bar chart from gain-based feature importance
    if not fi_map:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No importance data available", ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return path
    
    # Sort by importance and take top features
    items = sorted(fi_map.items(), key=lambda kv: kv[1], reverse=True)[:15]
    labels = [k.replace('_gbp', '') for k, _ in items]
    vals = [v for _, v in items]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.barh(range(len(labels)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f"{target} – Feature Importance (SHAP unavailable)")
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def write_all_figures(figout: str, summary: Dict) -> Dict[str, str]:
    """Generate all figures and save to output directory.
    
    Returns a dictionary mapping figure keys to their RELATIVE paths 
    (relative to the reports directory, for HTML embedding).
    """
    _ensure_dir(figout)
    
    # Compute relative path from reports dir to figout for HTML embedding
    # Expected structure: reports/assets/ -> relative path is "assets/"
    rel_prefix = os.path.basename(figout)  # typically "assets"
    
    paths = {}
    for res in summary.get("results", []):
        target = res.get("target", res.get("target"))
        
        # Holdout figure
        abs_path = save_holdout_figure(figout, target, res.get("holdout", {}))
        paths[f"{target}_holdout"] = f"{rel_prefix}/{os.path.basename(abs_path)}"
        
        # Single factor CV plots
        for g, rows in res.get("single_factor", {}).items():
            if not rows:
                continue
            r0 = rows[0]
            per = r0.get("per_fold", {})
            fl = per.get("losses", [])
            if fl:
                labels = [f"F{i+1}" for i in range(len(fl))]
                abs_path = save_cv_bar(figout, target, f"{g}_sf_top1", labels, fl)
                paths[f"{target}_{g}_sf"] = f"{rel_prefix}/{os.path.basename(abs_path)}"
        
        # Multi-factor best CV plot
        best = res.get("best_combo", {})
        per = best.get("per_fold", {}) if best else {}
        fl = per.get("losses", [])
        if fl:
            labels = [f"F{i+1}" for i in range(len(fl))]
            abs_path = save_cv_bar(figout, target, "mf_best", labels, fl)
            paths[f"{target}_mf"] = f"{rel_prefix}/{os.path.basename(abs_path)}"
        
        # Feature importance bar chart
        fi = res.get("feature_importance", {})
        abs_path = save_fi_bar(figout, target, fi)
        paths[f"{target}_fi"] = f"{rel_prefix}/{os.path.basename(abs_path)}"
        
        # Feature importance beeswarm (SHAP if available)
        shap_values = res.get("shap_values")
        shap_X = res.get("shap_X")
        shap_feature_names = res.get("shap_feature_names")
        abs_path = save_fi_beeswarm(
            figout, target, fi,
            shap_values=shap_values,
            shap_X=shap_X,
            shap_feature_names=shap_feature_names
        )
        paths[f"{target}_fi_beeswarm"] = f"{rel_prefix}/{os.path.basename(abs_path)}"
    # Save manifest (for debugging/reference)
    manifest_path = os.path.join(figout, "figures_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    
    return paths

