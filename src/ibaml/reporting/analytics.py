from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

def _ts() -> str:
    """Get current UTC timestamp in YYYYMMDDTHHMMSSZ format."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_target_name(res: Dict) -> str:
    """Get target name from result dict (supports both 'target' and legacy 'target')."""
    return res.get("target", res.get("target", "unknown"))


def create_tables(
    summary: Dict,
    full_group_columns: Dict[str, List[str]],
    outdir: str = "artifacts"
) -> Dict[str, str]:
    """
    Create performance and error analysis tables.
    
    Returns:
        Dict mapping table names to file paths
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}
    
    targets = []
    deltas = []
    gammas = []
    realized_sharpe = []
    predicted_sharpe = []
    simulated_sharpe = []
    realized_mdd = []
    predicted_mdd = []
    simulated_mdd = []
    realized_total_return = []
    predicted_total_return = []
    simulated_total_return = []
    cv_mse_vals = []
    cv_sign_acc_vals = []
    cv_loss_vals = []
    holdout_mse_vals = []
    holdout_loss_vals = []
    holdout_sign_acc_vals = []
    
    for res in summary.get("results", []):
        target = _get_target_name(res)
        targets.append(target)
        deltas.append(res.get("delta", -0.01))
        gammas.append(res.get("gamma", 0.01))
        
        metrics = res.get("holdout", {}).get("metrics", {})
        realized_sharpe.append(metrics.get("sharpe_realized", np.nan))
        predicted_sharpe.append(metrics.get("sharpe_pred", np.nan))
        simulated_sharpe.append(metrics.get("sharpe_sim", np.nan))
        realized_mdd.append(metrics.get("mdd_realized", np.nan))
        predicted_mdd.append(metrics.get("mdd_pred", np.nan))
        simulated_mdd.append(metrics.get("mdd_sim", np.nan))
        
        # Calculate cumulative returns from holdout data
        holdout = res.get("holdout", {})
        realized_arr = np.array(holdout.get("realized", []))
        predicted_arr = np.array(holdout.get("predicted", []))
        simulated_arr = np.array(holdout.get("simulated", []))
        
        # Use wealth indexing consistently: final value of cumprod(1 + r) - 1
        if len(realized_arr) > 0:
            realized_cum = float(np.cumprod(1.0 + realized_arr)[-1] - 1.0)
            realized_total_return.append(realized_cum)
        else:
            realized_total_return.append(np.nan)

        if len(predicted_arr) > 0:
            predicted_cum = float(np.cumprod(1.0 + predicted_arr)[-1] - 1.0)
            predicted_total_return.append(predicted_cum)
        else:
            predicted_total_return.append(np.nan)

        if len(simulated_arr) > 0:
            # Simulated is already cumulative returns array; take final value
            simulated_cum = float(simulated_arr[-1])
            simulated_total_return.append(simulated_cum)
        else:
            simulated_total_return.append(np.nan)
        
        best_combo = res.get("best_combo", {})
        cv_loss_vals.append(best_combo.get("mean_loss", np.nan))
        cv_sign_acc_vals.append(best_combo.get("mean_sign_acc", np.nan))
        # CV MSE (squared RMSE)
        cv_rmse = best_combo.get("mean_rmse", np.nan)
        cv_mse = (cv_rmse ** 2) if not np.isnan(cv_rmse) else np.nan
        cv_mse_vals.append(cv_mse)
        
        # Holdout metrics
        holdout_loss_vals.append(metrics.get("holdout_loss", np.nan))
        holdout_sign_acc_vals.append(metrics.get("sign_acc", np.nan))
        # Holdout MSE (squared RMSE)
        holdout_rmse = metrics.get("rmse", np.nan)
        holdout_mse = (holdout_rmse ** 2) if not np.isnan(holdout_rmse) else np.nan
        holdout_mse_vals.append(holdout_mse)
    
    # Table 1: Delta/Gamma per target
    table1 = pd.DataFrame({
        "Target": targets,
        "Delta (δ)": deltas,
        "Gamma (γ)": gammas,
    })
    table1_path = os.path.join(outdir, "Table1_DeltaGamma.csv")
    table1.to_csv(table1_path, index=False)
    results["table1_delta_gamma"] = table1_path
    
    # Table 2: Performance comparison
    table2 = pd.DataFrame({
        "Target": targets,
        "Realized Total Return": realized_total_return,
        "Predicted Total Return": predicted_total_return,
        "Simulated Total Return": simulated_total_return,
        "Realized Sharpe": realized_sharpe,
        "Predicted Sharpe": predicted_sharpe,
        "Simulated Sharpe": simulated_sharpe,
        "Realized MDD": realized_mdd,
        "Predicted MDD": predicted_mdd,
        "Simulated MDD": simulated_mdd,
    })
    table2_path = os.path.join(outdir, "Table2_Performance.csv")
    table2.to_csv(table2_path, index=False)
    results["table2_performance"] = table2_path
    
    # Table 3: Error analysis
    # Split between CV and Holdout metrics with MSE instead of RMSE
    table3 = pd.DataFrame({
        "Target": targets,
        "CV Loss": cv_loss_vals,
        "CV MSE": cv_mse_vals,
        "CV Sign Accuracy": cv_sign_acc_vals,
        "Holdout Loss": holdout_loss_vals,
        "Holdout MSE": holdout_mse_vals,
        "Holdout Sign Accuracy": holdout_sign_acc_vals,
    })
    table3_path = os.path.join(outdir, "Table3_ErrorAnalysis.csv")
    table3.to_csv(table3_path, index=False)
    results["table3_error_analysis"] = table3_path
    
    return results


def extract_gbp_formulas(
    summary: Dict,
    full_group_columns: Dict[str, List[str]],
    outdir: str = "artifacts",
    target_endg_columns: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Extract and document GBP formulas for each target's best model.
    
    Creates readable mathematical representations of feature combinations.
    """
    from ..utils.expressions import mask_expression
    
    os.makedirs(outdir, exist_ok=True)
    results = {}
    
    # Create detailed GBP document
    gbp_doc = ["# GBP Formulas - Best Models per Target\n"]
    gbp_doc.append(f"Generated: {_ts()}\n\n")
    
    gbp_json = {}
    
    for res in summary.get("results", []):
        target = _get_target_name(res)
        best = res.get("best_combo", {})
        
        if not best.get("groups"):
            continue
        
        gbp_doc.append(f"## {target}\n")
        
        # Store formula structure
        gbp_json[target] = {
            "target": target,
            "groups": best.get("groups", []),
            "components": [],
            "formula": "",
            "mean_loss": best.get("mean_loss", None),
            "mean_rmse": best.get("mean_rmse", None),
            "mean_sign_acc": best.get("mean_sign_acc", None),
        }
        
        components = []
        formulas = []
        
        for group, cols in zip(best.get("groups", []), best.get("columns", [])):
            # For ENDG, use TARGET-SPECIFIC columns only
            if group == "ENDG" and target_endg_columns:
                all_cols = target_endg_columns.get(target, [])
            else:
                all_cols = full_group_columns.get(group, [])
            
            if not all_cols:
                all_cols = ["endogenous"] if group == "ENDG" else []
            
            expr = mask_expression(all_cols, cols)
            components.append({
                "group": group,
                "features": cols,
                "formula": expr
            })
            formulas.append(f"{group}: {expr}")
            
            gbp_doc.append(f"### Group: {group}\n")
            gbp_doc.append(f"Selected features: {', '.join(cols)}\n\n")
            gbp_doc.append(f"GBP Formula:\n```\n{expr}\n```\n\n")
        
        # Combine formulas
        full_formula = " ∧ ".join(formulas)  # ∧ = separate groups
        gbp_json[target]["components"] = components
        gbp_json[target]["formula"] = full_formula
        
        gbp_doc.append(f"**Full Model Formula:**\n```\n{target}_IBA = {full_formula}\n```\n\n")
        gbp_doc.append(f"**Performance:**\n- Loss: {best.get('mean_loss', 'N/A')}\n")
        gbp_doc.append(f"- RMSE: {best.get('mean_rmse', 'N/A')}\n")
        gbp_doc.append(f"- Sign Accuracy: {best.get('mean_sign_acc', 'N/A')}\n\n")
    
    # Write markdown
    gbp_md_path = os.path.join(outdir, "GBP_Formulas.md")
    with open(gbp_md_path, "w") as f:
        f.write("\n".join(gbp_doc))
    results["gbp_formulas_md"] = gbp_md_path
    
    # Write JSON
    gbp_json_path = os.path.join(outdir, "GBP_Formulas.json")
    with open(gbp_json_path, "w") as f:
        json.dump(gbp_json, f, indent=2)
    results["gbp_formulas_json"] = gbp_json_path
    
    return results


def create_feature_importance_with_mapping(
    summary: Dict,
    outdir: str = "artifacts"
) -> Dict[str, str]:
    """
    Create feature importance data with feature name mappings.
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}
    
    feature_map = {
        # Bonds (B)
        "BAMLEMCBPIEY_CHG": "EM Corporate Bond Spread",
        "BAMLHE00EHYIEY_CHG": "High Yield OAS",
        "BAMLC0A1CAAAEY_CHG": "AAA Corporate Bond Spread",
        "BAMLH0A2HYBEY_CHG": "High Yield Bond Spread",
        # Interest Rates (IR)
        "T1YFF_CHG": "1Y Fed Funds Rate",
        "T10Y2Y_CHG": "10Y-2Y Yield Spread",
        "TB3MS_CHG": "3M T-Bill Rate",
        "T5YFF_CHG": "5Y Forward Fed Funds",
        # Volatility (V)
        "VIXCLS": "VIX Index",
        "EMVCOMMMKT": "EMV Commodities Market",
        "EMVFINCRISES": "EMV Financial Crises",
        "EMVMACRORE": "EMV Macro RE",
        # Asset Pricing (AP)
        "Mkt_RF": "Market Risk Premium",
        "SMB": "Small-Minus-Big",
        "HML": "High-Minus-Low",
        "MSCIEM": "MSCI Emerging Markets",
        # Trend Following (TF)
        "PTFSBD": "Bond Trend Following",
        "PTFSCOM": "Commodity Trend Following",
        "PTFSIR": "Interest Rate Trend Following",
        "PTFSSTK": "Stock Trend Following",
    }
    
    fi_json = {}
    
    for res in summary.get("results", []):
        target = _get_target_name(res)
        fi_raw = res.get("feature_importance", {})
        
        fi_mapped = []
        for feat, importance in sorted(fi_raw.items(), key=lambda x: x[1], reverse=True):
            # Extract base feature name
            parts = feat.split("_")
            base_name = "_".join(parts[:-2]) if len(parts) > 2 else feat
            
            mapped_name = feature_map.get(base_name, base_name)
            window = parts[-2] if len(parts) > 1 else "N/A"
            
            fi_mapped.append({
                "feature": feat,
                "mapped_name": mapped_name,
                "window": window,
                "importance": importance
            })
        
        fi_json[target] = fi_mapped
    
    fi_json_path = os.path.join(outdir, "FeatureImportance_Mapped.json")
    with open(fi_json_path, "w") as f:
        json.dump(fi_json, f, indent=2)
    results["feature_importance_mapped"] = fi_json_path
    
    return results


def write_analytical_report_html(
    outdir: str,
    summary: Dict,
    tables_dir: Dict[str, str],
    gbp_dir: Dict[str, str],
    fig_manifest: Optional[Dict[str, str]] = None,
    full_group_columns: Optional[Dict[str, List[str]]] = None,
    target_endg_columns: Optional[Dict[str, List[str]]] = None,
    config_yaml: Optional[str] = None
) -> str:
    """
    Create analytical HTML report with embedded tables, formulas, figures, and detailed target analysis.
    """
    from ..utils.expressions import mask_expression
    
    os.makedirs(outdir, exist_ok=True)
    
    css = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 20px;
        background: #f5f5f5;
        color: #333;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1a5490;
        border-bottom: 2px solid #1a5490;
        padding-bottom: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }
    th {
        background: #1a5490;
        color: white;
        padding: 12px;
        text-align: left;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #ddd;
    }
    tr:hover {
        background: #f9f9f9;
    }
    .card {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 15px;
        margin: 15px 0;
        background: #fafafa;
    }
    .formula {
        background: #f0f0f0;
        padding: 10px;
        border-left: 4px solid #1a5490;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    .metric {
        display: inline-block;
        background: #e8f4f8;
        padding: 8px 12px;
        margin: 5px;
        border-radius: 4px;
        font-size: 12px;
    }
    img {
        max-width: 100%;
        height: auto;
        margin: 20px 0;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    """
    
    html = ["<!DOCTYPE html>", "<html>", "<head>"]
    html.append("<meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("<title>IBAML Report</title>")
    html.append(f"<style>{css}</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")
    
    # Header
    html.append("<h1>IBAML Report</h1>")
    html.append(f"<p><strong>Generated</strong>: {_ts()}</p>")
    html.append("<p><strong>Analysis</strong>: Performance Forecasting and signal-gated investment simulation based on Interpolative Boolean Algebra Feature Reduction.</p>")
    
    # Executive Summary
    html.append("<h2>Executive Summary</h2>")
    html.append("<p>Analysis across 8 hedge fund targets indices 3 months ahead performance forecasting using signal-gated simulation logic.</p>")
    html.append("<ul>")
    html.append("<li><strong>Time Period</strong>: 2022-08 to 2025-07 (holdout)</li>")
    # Load targets from config_yaml if provided
    import yaml
    if config_yaml:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)
        targets_list = config.get("dataset", {}).get("target_columns", [])
        targets_str = ", ".join(targets_list)
        html.append(f"<li><strong>Strategies</strong>: {len(targets_list)} ({targets_str})</li>")
    else:
        html.append("<li><strong>Strategies</strong>: 8 (CTA, CA, DS, ELS, EMN, EDMS, FIA, GM)</li>")
    html.append("<li><strong>Models</strong>: Quantile-specific feature combinations using IBA GBP methodology</li>")
    html.append("</ul>")
    
    # Table 1
    html.append("<h2>Table 1: Delta/Gamma Thresholds</h2>")
    if "table1_delta_gamma" in tables_dir:
        df = pd.read_csv(tables_dir["table1_delta_gamma"])
        html.append("<table>")
        html.append("<tr>")
        for col in df.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr>")
        for _, row in df.iterrows():
            html.append("<tr>")
            for val in row:
                html.append(f"<td>{val}</td>")
            html.append("</tr>")
        html.append("</table>")
    
    # Table 2
    html.append("<h2>Table 2: Performance Metrics</h2>")
    if "table2_performance" in tables_dir:
        df = pd.read_csv(tables_dir["table2_performance"])
        html.append("<table>")
        html.append("<tr>")
        for col in df.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr>")
        for _, row in df.iterrows():
            html.append("<tr>")
            for i, val in enumerate(row):
                if isinstance(val, (int, float)) and i > 0:
                    html.append(f"<td>{val:.4f}</td>")
                else:
                    html.append(f"<td>{val}</td>")
            html.append("</tr>")
        html.append("</table>")
    
    # Table 3
    html.append("<h2>Table 3: Error Analysis</h2>")
    if "table3_error_analysis" in tables_dir:
        df = pd.read_csv(tables_dir["table3_error_analysis"])
        html.append("<table>")
        html.append("<tr>")
        for col in df.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr>")
        for _, row in df.iterrows():
            html.append("<tr>")
            for i, val in enumerate(row):
                if isinstance(val, (int, float)) and i > 0:
                    html.append(f"<td>{val:.6f}</td>")
                else:
                    html.append(f"<td>{val}</td>")
            html.append("</tr>")
        html.append("</table>")
    
    # GBP Formulas and Figures - target Models
    html.append("<h2>IBA ML Models</h2>")
    html.append("<p>Detailed analysis of optimal Generalized Boolean Polynomial combinations per target with holdout performance evaluation.</p>")
    
    if not full_group_columns:
        # Load factor groups from config_yaml if provided
        if config_yaml:
            with open(config_yaml, "r") as f:
                config = yaml.safe_load(f)
            factors = config.get("factors", {})
            windows = config.get("dataset", {}).get("zscore_windows", [6, 12])
            full_group_columns = {}
            for group, base_factors in factors.items():
                cols = []
                for factor in base_factors:
                    for win in windows:
                        cols.append(f"{factor}_{{win}}m_zs".replace("{win}", str(win)))
                full_group_columns[group] = cols
            full_group_columns["ENDG"] = ["endogenous"]
        else:
            full_group_columns = {
                "ENDG": ["endogenous"],
                "IR": ["T1YFF_CHG_6m_zs", "T1YFF_CHG_12m_zs", "T10Y2Y_CHG_6m_zs",
                       "T10Y2Y_CHG_12m_zs", "TB3MS_CHG_6m_zs", "TB3MS_CHG_12m_zs",
                       "T5YFF_CHG_6m_zs", "T5YFF_CHG_12m_zs"],
                "B": ["BAMLEMCBPIEY_CHG_6m_zs", "BAMLEMCBPIEY_CHG_12m_zs", "BAMLHE00EHYIEY_CHG_6m_zs",
                      "BAMLHE00EHYIEY_CHG_12m_zs", "BAMLC0A1CAAAEY_CHG_6m_zs", "BAMLC0A1CAAAEY_CHG_12m_zs",
                      "BAMLH0A2HYBEY_CHG_6m_zs", "BAMLH0A2HYBEY_CHG_12m_zs"],
                "V": ["VIXCLS_6m_zs", "VIXCLS_12m_zs", "EMVCOMMMKT_6m_zs", "EMVCOMMMKT_12m_zs",
                      "EMVFINCRISES_6m_zs", "EMVFINCRISES_12m_zs", "EMVMACRORE_6m_zs", "EMVMACRORE_12m_zs"],
                "AP": ["Mkt_RF_6m_zs", "Mkt_RF_12m_zs", "SMB_6m_zs", "SMB_12m_zs",
                       "HML_6m_zs", "HML_12m_zs", "MSCIEM_6m_zs", "MSCIEM_12m_zs"],
                "TF": ["PTFSBD_6m_zs", "PTFSBD_12m_zs", "PTFSCOM_6m_zs", "PTFSCOM_12m_zs",
                       "PTFSIR_6m_zs", "PTFSIR_12m_zs", "PTFSSTK_6m_zs", "PTFSSTK_12m_zs"],
            }
    
    for res in summary.get("results", []):
        target = _get_target_name(res)
        best = res.get("best_combo", {})
        holdout = res.get("holdout", {})
        metrics = holdout.get("metrics", {})
        
        if not best.get("groups"):
            continue
        
        holdout_arr = holdout.get("simulated", [])
        cum_return = float(holdout_arr[-1]) if holdout_arr else np.nan
        
        html.append(f"<div class='card'>")
        html.append(f"<h3>{target} Target</h3>")
        
        # IBA Model formula
        formulas = []
        for group, cols in zip(best.get("groups", []), best.get("columns", [])):
            # For ENDG, use target-SPECIFIC columns only
            if group == "ENDG" and target_endg_columns:
                all_cols = target_endg_columns.get(target, [])
            else:
                all_cols = full_group_columns.get(group, []) if full_group_columns else []
            
            if not all_cols:
                all_cols = ["endogenous"] if group == "ENDG" else []
            
            expr = mask_expression(all_cols, cols)
            formulas.append(f"{group}: {expr}")
        
        full_formula = " ∧ ".join(formulas)
        html.append(f"<h4>IBA Model</h4>")
        html.append(f"<div class='formula'>{target}_IBA = {full_formula}</div>")
        
        # Helper to format values safely
        def fmt_val(val, fmt_spec=".6f"):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{val:{fmt_spec}}"
            except Exception:
                return "N/A"
        
        def fmt_pct(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{val:.2%}"
            except Exception:
                return "N/A"
        
        # Performance table
        html.append(f"<h4>Performance Summary</h4>")
        html.append(f"<table style='font-size: 12px;'>")
        html.append(f"<tr><th>Metric</th><th>Cross-Validation</th><th>Holdout</th></tr>")
        
        cv_rmse = best.get('mean_rmse')
        cv_mse = (cv_rmse ** 2) if cv_rmse is not None and not np.isnan(cv_rmse) else np.nan
        ho_rmse = metrics.get('rmse')
        ho_mse = (ho_rmse ** 2) if ho_rmse is not None and not np.isnan(ho_rmse) else np.nan
        
        html.append(f"<tr><td>Loss</td><td>{fmt_val(best.get('mean_loss'))}</td><td>{fmt_val(metrics.get('holdout_loss'))}</td></tr>")
        html.append(f"<tr><td>MSE</td><td>{fmt_val(cv_mse)}</td><td>{fmt_val(ho_mse)}</td></tr>")
        html.append(f"<tr><td>Sign Accuracy</td><td>{fmt_pct(best.get('mean_sign_acc'))}</td><td>{fmt_pct(metrics.get('sign_acc'))}</td></tr>")
        html.append(f"<tr><td>Cumulative Return</td><td>-</td><td>{fmt_pct(cum_return)}</td></tr>")
        html.append(f"<tr><td>Sharpe Ratio</td><td>-</td><td>{fmt_val(metrics.get('sharpe_sim'), '.4f')}</td></tr>")
        html.append(f"<tr><td>Max Drawdown</td><td>-</td><td>{fmt_pct(metrics.get('mdd_sim'))}</td></tr>")
        html.append(f"</table>")
        
        # Add hyperparameters if available
        hyperopt = res.get("hyperopt", {})
        if hyperopt and hyperopt.get("best_params"):
            html.append(f"<h4>Optimized Hyperparameters</h4>")
            html.append(f"<table style='font-size: 12px;'>")
            html.append(f"<tr><th>Parameter</th><th>Value</th></tr>")
            for param, val in hyperopt["best_params"].items():
                if isinstance(val, float):
                    html.append(f"<tr><td>{param}</td><td>{val:.6f}</td></tr>")
                else:
                    html.append(f"<tr><td>{param}</td><td>{val}</td></tr>")
            html.append(f"</table>")
            html.append(f"<p><em>Best CV loss: {fmt_val(hyperopt.get('best_value'), '.6f')} ({hyperopt.get('n_trials', 0)} trials)</em></p>")
        
        # Add delta/gamma thresholds
        delta = res.get("delta", np.nan)
        gamma = res.get("gamma", np.nan)
        html.append(f"<p><strong>Thresholds:</strong> δ = {fmt_val(delta)}, γ = {fmt_val(gamma)}</p>")
        
        # Add figures if available
        if fig_manifest:
            if f"{target}_holdout" in fig_manifest:
                fig_path = fig_manifest[f"{target}_holdout"]
                html.append(f"<h4>Holdout Performance (2023-2024)</h4>")
                html.append(f"<p><em>Realized, Predicted, and Simulated cumulative returns over test period</em></p>")
                html.append(f"<img src='{fig_path}' alt='{target} holdout' style='max-width: 800px;'/>")
            
            if f"{target}_mf" in fig_manifest:
                fig_path = fig_manifest[f"{target}_mf"]
                html.append(f"<h4>Multi-Factor Search - CV Losses</h4>")
                html.append(f"<p><em>Loss distribution across feature combinations during cross-validation</em></p>")
                html.append(f"<img src='{fig_path}' alt='{target} MF' style='max-width: 800px;'/>")
            
            if f"{target}_fi" in fig_manifest:
                fig_path = fig_manifest[f"{target}_fi"]
                html.append(f"<h4>Feature Importance (Top 20)</h4>")
                html.append(f"<p><em>XGBoost feature importance from best model</em></p>")
                html.append(f"<img src='{fig_path}' alt='{target} FI' style='max-width: 800px;'/>")
            
            if f"{target}_fi_beeswarm" in fig_manifest:
                fig_path = fig_manifest[f"{target}_fi_beeswarm"]
                html.append(f"<h4>Feature Importance by Group (Beeswarm)</h4>")
                html.append(f"<p><em>Feature importance distribution across feature groups showing which group drives predictions</em></p>")
                html.append(f"<img src='{fig_path}' alt='{target} FI Beeswarm' style='max-width: 800px;'/>")
        
        html.append(f"</div>")
    
    # Methodology
    html.append("<h2>Methodology</h2>")
    
    html.append("<h3>Feature Reduction</h3>")
    html.append("<p>Interpolative Boolean Algebra (IBA) with Generalized Boolean Polynomials (GBP).</p>")

    html.append("<h3>Multi-Factor Search</h3>")
    html.append("<ul>")
    html.append("<li><strong>target</strong>: Exhaustive search of 2<sup>N</sup> - 1 valid feature combinations per target</li>")
    html.append("<li><strong>Selection Criterion</strong>: Minimum 4-component weighted loss (SE + ME<sub>neg</sub> + ME<sub>pos</sub> + MSE)</li>")
    html.append("<li><strong>Weighting</strong>: λ₁ - sign error, λ₂ - positive magnitude, λ₃ - negative magnitude</li>")
    html.append("</ul>")
    html.append("<h3>Cross-Validation</h3>")
    html.append("<ul>")
    html.append("<li><strong>Scheme</strong>: Expanding window with K folds</li>")
    html.append("<li><strong>Evaluation</strong>: Custom objective function for fold ranking</li>")
    html.append("</ul>")
    html.append("<h3>Holdout Evaluation</h3>")
    html.append("<ul>")
    html.append("<li><strong>Metrics</strong>: Loss, MSE, sign accuracy, Sharpe ratio (annualized), maximum drawdown</li>")
    html.append("<li><strong>Simulation</strong>: Signal-gated trading (long only when prediction ≥ 0)</li>")
    html.append("</ul>")
    html.append("</div>")
    html.append("</body>")
    html.append("</html>")
    
    html_path = os.path.join(outdir, f"AnalyticalReport_{_ts()}.html")
    with open(html_path, "w") as f:
        f.write("\n".join(html))
    
    return html_path
