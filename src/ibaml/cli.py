from __future__ import annotations
import os
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd
from joblib import Parallel, delayed

from .config.schemas import load_config, Config
from .data.prep import read_raw_pair, cumulative_shifted_returns, normalize_to_month_end
from .features.engineering import build_group_features
from .losses.objective import get_quantile_params_from_config
from .pipelines.target_runner import run_target_pipeline
from .reporting.export_csv import write_results_csvs
from .reporting.figures import write_all_figures
from .reporting.analytics import (
    create_tables, extract_gbp_formulas, create_feature_importance_with_mapping,
    write_analytical_report_html
)

logger = logging.getLogger(__name__)


def ts_utc() -> str:
    """Generate UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _set_thread_env():
    """Set environment variables to prevent thread oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("XGBOOST_NUM_THREADS", "1")


def run_all_targets(
    cfg: Config,
    xgb_overrides: Optional[Dict] = None,
    top_k: int = 1,
    limit_mask_size: Optional[int] = None,
    allow_empty_groups: bool = True,
    min_groups_in_combo: int = 1,
    max_groups_in_combo: Optional[int] = None,
    n_jobs_targets: int = 1,
    n_jobs_groups: int = 1,
    n_jobs_masks: int = 1,
    n_jobs_combos: int = 1,
    outdir: str = "artifacts",
    models_subdir: str = "models",
    html: bool = True,
) -> Dict:
    """Run the complete IBAML pipeline for all targets.
    
    Args:
        cfg: Configuration object
        xgb_overrides: XGBoost parameter overrides
        top_k: Number of top single-factor models per group
        limit_mask_size: Maximum mask size for feature selection
        allow_empty_groups: Allow empty groups in combinations
        min_groups_in_combo: Minimum groups in combination
        max_groups_in_combo: Maximum groups in combination
        n_jobs_targets: Parallel jobs for targets (formerly strategies)
        n_jobs_groups: Parallel jobs for factor groups
        n_jobs_masks: Parallel jobs for feature masks
        n_jobs_combos: Parallel jobs for combinations
        outdir: Output directory
        models_subdir: Subdirectory for model files
        html: Generate HTML report
        
    Returns:
        Summary dictionary with all results
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Load and preprocess data
    df_f, df_t = read_raw_pair(
        cfg.dataset.factors_path, cfg.dataset.targets_path, cfg.dataset.date_column,
        cfg.dataset.parse_dates, align_freq=cfg.dataset.align_freq,
        align_agg=cfg.dataset.align_agg, align_how=cfg.dataset.align_how
    )
    
    # Normalize to month-end
    df_f.index = normalize_to_month_end(pd.DatetimeIndex(df_f.index))
    df_t.index = normalize_to_month_end(pd.DatetimeIndex(df_t.index))
    
    # Compute cumulative shifted returns
    cum = cumulative_shifted_returns(
        df_t[cfg.dataset.target_columns],
        cfg.dataset.cumulative_horizon,
        cfg.dataset.forecast_shift
    )
    cum.index = normalize_to_month_end(pd.DatetimeIndex(cum.index))
    
    # Build factor group features
    groups_frames_all = build_group_features(df_f, cfg.factors, cfg.dataset.zscore_windows)
    full_group_columns = {g: list(df.columns) for g, df in groups_frames_all.items()}
    
    # Prepare XGBoost parameters
    xgb_params = dict((cfg.xgb.params if cfg.xgb else {}))
    if xgb_overrides:
        xgb_params.update(xgb_overrides)
    num_boost_round = int(xgb_params.pop("n_estimators", 500))
    xgb_params.setdefault("nthread", 1)
    xgb_params.setdefault("verbosity", 0)
    xgb_params.setdefault("objective", "reg:squarederror")
    
    # Get quantile parameters
    config_dict = cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg.__dict__
    quantile_delta, quantile_gamma = get_quantile_params_from_config(config_dict)
    
    targets = cfg.dataset.target_columns
    
    def _process_target(target: str) -> Dict:
        """Process a single target through the pipeline."""
        y_all = cum[target].astype("float32")
        result = run_target_pipeline(
            target=target,
            y_all=y_all,
            df_targets=df_t,
            group_frames_all=groups_frames_all,
            cfg=cfg,
            xgb_params=xgb_params,
            num_boost_round=num_boost_round,
            quantile_delta=quantile_delta,
            quantile_gamma=quantile_gamma,
            config_dict=config_dict,
            outdir=outdir,
            models_subdir=models_subdir,
            top_k=top_k,
            limit_mask_size=limit_mask_size,
            allow_empty_groups=allow_empty_groups,
            min_groups_in_combo=min_groups_in_combo,
            max_groups_in_combo=max_groups_in_combo,
            n_jobs_groups=n_jobs_groups,
            n_jobs_masks=n_jobs_masks,
            n_jobs_combos=n_jobs_combos,
        )
        return result.to_dict()
    
    # Run pipeline for all targets
    if n_jobs_targets and n_jobs_targets != 1:
        res = Parallel(n_jobs=n_jobs_targets, backend="loky", prefer="processes")(
            delayed(_process_target)(t) for t in targets
        )
    else:
        res = [_process_target(t) for t in targets]
    
    # Build summary
    summary = {
        "config": {
            "splits": cfg.splits.model_dump(),
            "objective": cfg.objective.model_dump(),
            "zscore_windows": cfg.dataset.zscore_windows,
            "cumulative_horizon": cfg.dataset.cumulative_horizon,
            "forecast_shift": cfg.dataset.forecast_shift,
        },
        "results": res,
        "full_group_columns": full_group_columns,
    }
    
    # Generate outputs
    _generate_outputs(cfg, summary, full_group_columns, outdir)
    
    return summary


def _generate_outputs(
    cfg: Config,
    summary: Dict,
    full_group_columns: Dict[str, List[str]],
    outdir: str,
) -> None:
    """Generate all output files: CSVs, figures, reports."""
    # Reload factors for column mapping
    df_f2, _ = read_raw_pair(
        cfg.dataset.factors_path, cfg.dataset.targets_path, cfg.dataset.date_column,
        cfg.dataset.parse_dates, cfg.dataset.align_freq, cfg.dataset.align_agg,
        cfg.dataset.align_how
    )
    colmap = {
        g: list(df.columns)
        for g, df in build_group_features(df_f2, cfg.factors, cfg.dataset.zscore_windows).items()
    }
    
    # Create output directories
    data_dir = os.path.join(outdir, "data")
    reports_dir = os.path.join(outdir, "reports")
    assets_dir = os.path.join(reports_dir, "assets")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Write CSVs
    write_results_csvs(data_dir, summary, colmap)
    
    # Generate figures
    fig_manifest = write_all_figures(assets_dir, summary)
    
    # Clean summary for JSON (remove large arrays)
    summary_clean = json.loads(json.dumps(summary))
    for res in summary_clean.get("results", []):
        res.pop("leaderboard", None)
        res.pop("single_factor", None)
        # Remove large SHAP objects to keep JSON compact
        res.pop("shap_values", None)
        res.pop("shap_X", None)
    
    run_json = os.path.join(reports_dir, f"run_all_{ts_utc()}.json")
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(summary_clean, f, indent=2)
    
    # Generate analytics tables
    tables_dict = create_tables(summary, colmap, reports_dir)
    
    # Build target-specific ENDG columns
    target_endg_columns = {}
    for res in summary.get("results", []):
        target = res.get("target", res.get("target"))
        target_endg_columns[target] = [
            f"{target}_raw",
            f"{target}_6m_zs",
            f"{target}_12m_zs"
        ]
    
    gbp_info = extract_gbp_formulas(summary, full_group_columns, reports_dir, target_endg_columns)
    create_feature_importance_with_mapping(summary, reports_dir)
    
    # Generate reports with tables
    html_report = write_analytical_report_html(
        reports_dir, summary, tables_dict, gbp_info,
        full_group_columns=full_group_columns,
        target_endg_columns=target_endg_columns,
        fig_manifest=fig_manifest
    )
    
    logger.info(f"Results saved:")
    logger.info(f"  JSON: {run_json}")
    logger.info(f"  HTML Report: {html_report}")
    
    print(run_json)
    print(html_report)


def main():
    """Main entry point for the CLI."""
    _set_thread_env()
    
    parser = argparse.ArgumentParser(
        description="IBAML: Investment Target Forecasting with IBA + XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("config", help="Path to YAML configuration file")
    
    # XGBoost overrides
    parser.add_argument("--xgb", type=str, default=None,
                        help="JSON string with XGBoost parameter overrides")
    
    # Search parameters
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of top single-factor models per group")
    parser.add_argument("--limit-mask-size", type=int, default=0,
                        help="Maximum mask size (0=unlimited)")
    parser.add_argument("--allow-empty-groups", action="store_true", default=True,
                        help="Allow empty groups in combinations")
    parser.add_argument("--min-groups-in-combo", type=int, default=1,
                        help="Minimum groups in multi-factor combination")
    parser.add_argument("--max-groups-in-combo", type=int, default=0,
                        help="Maximum groups in combination (0=auto)")
    
    # Parallelization
    parser.add_argument("--n-jobs-strategies", "--n-jobs-targets", type=int, default=1,
                        dest="n_jobs_targets",
                        help="Parallel jobs across targets/strategies")
    parser.add_argument("--n-jobs-groups", type=int, default=1,
                        help="Parallel jobs across factor groups")
    parser.add_argument("--n-jobs-masks", type=int, default=1,
                        help="Parallel jobs across feature masks")
    parser.add_argument("--n-jobs-combos", type=int, default=1,
                        help="Parallel jobs across combinations")
    
    # Output
    parser.add_argument("--outdir", type=str, default="artifacts",
                        help="Output directory")
    parser.add_argument("--models-subdir", type=str, default="models",
                        help="Subdirectory for model files")
    parser.add_argument("--html", action="store_true", default=False,
                        help="Generate HTML report")
    
    # Other options
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--no-progress", action="store_true", default=False,
                        help="Disable progress bars")

    # Allow running a subset of targets
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated list of targets to run (overrides config.target_columns)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )
    
    # Load configuration
    cfg = load_config(args.config)

    # If user supplied --targets, override list in configuration
    if args.targets:
        # Split comma-separated list and strip whitespace
        cfg.dataset.target_columns = [t.strip() for t in args.targets.split(",") if t.strip()]
    
    # Handle zero values
    limit_mask_size = args.limit_mask_size if args.limit_mask_size != 0 else None
    max_groups_in_combo = args.max_groups_in_combo if args.max_groups_in_combo != 0 else None
    
    # Use config values as defaults
    search_cfg = cfg.search if cfg.search else {}
    allow_empty_groups = args.allow_empty_groups or getattr(search_cfg, 'allow_empty_groups', False)
    top_k = args.top_k or getattr(search_cfg, 'top_k', 1)
    limit_mask_size = limit_mask_size or getattr(search_cfg, 'limit_mask_size', None)
    min_groups_in_combo = args.min_groups_in_combo or getattr(search_cfg, 'min_groups_in_combo', 1)
    max_groups_in_combo = max_groups_in_combo or getattr(search_cfg, 'max_groups_in_combo', None)
    
    
    # Parse XGBoost overrides
    xgb_overrides = json.loads(args.xgb) if args.xgb else None
    
    # Run pipeline
    run_all_targets(
        cfg=cfg,
        xgb_overrides=xgb_overrides,
        top_k=top_k,
        limit_mask_size=limit_mask_size,
        allow_empty_groups=allow_empty_groups,
        min_groups_in_combo=min_groups_in_combo,
        max_groups_in_combo=max_groups_in_combo,
        n_jobs_targets=args.n_jobs_targets,
        n_jobs_groups=args.n_jobs_groups,
        n_jobs_masks=args.n_jobs_masks,
        n_jobs_combos=args.n_jobs_combos,
        outdir=args.outdir,
        models_subdir=args.models_subdir,
        html=args.html,
    )


if __name__ == "__main__":
    main()

