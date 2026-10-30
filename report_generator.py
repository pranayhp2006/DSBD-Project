"""
Report Generation and Export Module
Facilitates formatted executive HTML/Markdown reports and CSV bundle downloads
for Placement Cell analytics and student advisory dossiers.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any


def generate_placement_executive_summary_html(
    df: pd.DataFrame,
    model_comparison_df: pd.DataFrame,
    cluster_profiles: list,
    best_model_name: str
) -> str:
    """Generates an executive, printable HTML placement analytics report."""
    total_students = len(df)
    placed_count = int((df["Placement_Status"] == "Placed").sum()) if "Placement_Status" in df.columns else 0
    placement_rate = round((placed_count / total_students) * 100, 1) if total_students > 0 else 0.0

    placed_df = df[df["Placement_Status"] == "Placed"] if "Placement_Status" in df.columns else df
    avg_package = round(float(placed_df["Salary_Package_LPA"].mean()), 2) if "Salary_Package_LPA" in placed_df.columns else 0.0
    max_package = round(float(placed_df["Salary_Package_LPA"].max()), 2) if "Salary_Package_LPA" in placed_df.columns else 0.0

    now_str = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # Model table rows
    model_rows_html = ""
    if model_comparison_df is not None and not model_comparison_df.empty:
        for _, row in model_comparison_df.iterrows():
            model_rows_html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #e2e8f0; font-weight: 600;">{row.get('Algorithm', '')}</td>
                <td style="padding: 8px; border: 1px solid #e2e8f0; text-align: center;">{row.get('Accuracy (%)', '')}%</td>
                <td style="padding: 8px; border: 1px solid #e2e8f0; text-align: center;">{row.get('Precision (%)', '')}%</td>
                <td style="padding: 8px; border: 1px solid #e2e8f0; text-align: center;">{row.get('Recall (%)', '')}%</td>
                <td style="padding: 8px; border: 1px solid #e2e8f0; text-align: center; color: #2563eb; font-weight: bold;">{row.get('F1-Score (%)', '')}%</td>
                <td style="padding: 8px; border: 1px solid #e2e8f0; text-align: center;">{row.get('ROC-AUC', '')}</td>
            </tr>
            """

    # Cluster profiles
    cluster_rows_html = ""
    for cp in cluster_profiles:
        recs = "".join(f"<li>{r}</li>" for r in cp.get("recommendations", []))
        metrics = cp.get("metrics", {})
        cluster_rows_html += f"""
        <div style="background: #f8fafc; border-left: 4px solid {cp.get('badge_color', '#3b82f6')}; padding: 12px; margin-bottom: 12px; border-radius: 4px;">
            <h4 style="margin: 0 0 6px 0; color: #1e293b;">{cp.get('tier_label', '')} <span style="font-size: 13px; font-weight: normal; color: #64748b;">({metrics.get('count', 0)} students)</span></h4>
            <p style="margin: 0 0 8px 0; font-size: 13px; color: #475569;"><strong>Target Focus:</strong> {cp.get('focus_area', '')}</p>
            <p style="margin: 0 0 6px 0; font-size: 13px; color: #475569;">
                Avg CGPA: <strong>{metrics.get('cgpa_mean', 0)}</strong> | 
                Avg Coding: <strong>{metrics.get('coding_mean', 0)}/100</strong> | 
                Avg Aptitude: <strong>{metrics.get('apt_mean', 0)}/100</strong> | 
                Placement Rate: <strong>{metrics.get('placed_pct', 0)}%</strong>
            </p>
            <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #334155;">{recs}</ul>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Placement Analytics & Machine Learning Evaluation Dossier</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #1e293b; background: #ffffff; line-height: 1.5; }}
            .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; }}
            .title {{ font-size: 26px; font-weight: 800; color: #0f172a; margin: 0; }}
            .subtitle {{ font-size: 14px; color: #64748b; margin-top: 4px; }}
            .grid {{ display: flex; gap: 16px; margin-bottom: 24px; }}
            .card {{ flex: 1; background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0; }}
            .metric-val {{ font-size: 28px; font-weight: 700; color: #2563eb; margin: 0; }}
            .metric-lbl {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-top: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
            th {{ background: #f8fafc; border: 1px solid #cbd5e1; padding: 10px; text-align: left; font-weight: 700; color: #334155; }}
            h3 {{ border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; color: #1e293b; margin-top: 28px; }}
            .badge {{ background: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1 class="title">Placement Analysis & Intelligence System</h1>
            <p class="subtitle">Comprehensive Machine Learning & Curriculum Alignment Report | Generated on {now_str}</p>
        </div>

        <div class="grid">
            <div class="card">
                <p class="metric-val">{total_students}</p>
                <p class="metric-lbl">Total Students Analyzed</p>
            </div>
            <div class="card">
                <p class="metric-val">{placement_rate}%</p>
                <p class="metric-lbl">Overall Placement Ratio</p>
            </div>
            <div class="card">
                <p class="metric-val">{avg_package} LPA</p>
                <p class="metric-lbl">Average Salary Package</p>
            </div>
            <div class="card">
                <p class="metric-val">{max_package} LPA</p>
                <p class="metric-lbl">Highest Recorded Package</p>
            </div>
        </div>

        <h3>1. Machine Learning Classification Performance Benchmark (CO2 & CO4)</h3>
        <p style="font-size: 13px; color: #475569;">
            Benchmarked against <strong>Logistic Regression, KNN, Decision Tree, and Support Vector Machine</strong>. 
            Selected Best Performing Model: <span class="badge">{best_model_name}</span>
        </p>
        <table>
            <thead>
                <tr>
                    <th>Algorithm</th>
                    <th style="text-align: center;">Accuracy</th>
                    <th style="text-align: center;">Precision</th>
                    <th style="text-align: center;">Recall</th>
                    <th style="text-align: center;">F1-Score</th>
                    <th style="text-align: center;">ROC-AUC</th>
                </tr>
            </thead>
            <tbody>
                {model_rows_html}
            </tbody>
        </table>

        <h3>2. Student Segmentation & K-Means Clustering Tiers (CO3)</h3>
        <p style="font-size: 13px; color: #475569;">
            Unsupervised clustering grouped the cohort into actionable performance tiers with targeted intervention blueprints:
        </p>
        {cluster_rows_html}

        <h3>3. Institutional Recommendations for Placement Cell</h3>
        <ul style="font-size: 13px; color: #334155; line-height: 1.8;">
            <li>Conduct focused Data Structures & System Design bootcamps for candidates approaching Tier 1 readiness.</li>
            <li>Enforce mandatory quantitative reasoning diagnostic tests every fortnight.</li>
            <li>Target 100% internship compliance before the commencement of pre-placement talks (Semester 7).</li>
            <li>Schedule specialized alumni mock interview cycles for students identified in Tier 3.</li>
        </ul>
    </body>
    </html>
    """
    return html
