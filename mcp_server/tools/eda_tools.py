import os

import numpy as np
import pandas as pd
import plotly.express as px

# Define directories for saving artifacts
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def run_eda(file_path: str) -> dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on a CSV dataset.
    Generates interactive Plotly visualizations and returns a structured dictionary.

    The analysis includes:
      1. BASIC INFO: Rows, columns, file size, column types
      2. MISSING VALUES: Count, percentage, and high-missing (>30%) warnings
      3. STATISTICAL SUMMARY: Mean, median, std, min, max for numeric; top 5 value counts for categorical
      4. PATTERNS & INSIGHTS: Auto target detection, strong correlations (>=0.7), IQR outliers, duplicate rows
      5. VISUALIZATIONS: Plotly correlation heatmap, numeric distribution plots saved as HTML
      6. WARNINGS: Heavy target imbalance (>80%), unique ID column flags, constant value flags

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        A structured dictionary containing basic info, missing values, statistics,
        patterns, warnings, and generated visualization paths.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found at: {file_path}")

    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # 1. BASIC INFO
        rows, cols = df.shape
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        basic_info = {
            "rows": int(rows),
            "columns": int(cols),
            "file_size_mb": round(file_size_mb, 4),
            "column_types": column_types
        }

        # 2. MISSING VALUES
        null_counts = df.isnull().sum().to_dict()
        null_percentages = {col: round(float((count / rows) * 100), 2) for col, count in null_counts.items()} if rows > 0 else {}

        missing_warnings = []
        for col, pct in null_percentages.items():
            if pct > 30.0:
                missing_warnings.append(
                    f"Column '{col}' has {pct}% missing values (exceeds 30% limit)."
                )

        missing_values = {
            "counts": null_counts,
            "percentages": null_percentages,
            "warnings": missing_warnings
        }

        # 3. STATISTICAL SUMMARY
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_summary = {}
        for col in numeric_cols:
            non_null_df = df[col].dropna()
            if not non_null_df.empty:
                numeric_summary[col] = {
                    "mean": round(float(non_null_df.mean()), 4),
                    "median": round(float(non_null_df.median()), 4),
                    "std": round(float(non_null_df.std()), 4) if len(non_null_df) > 1 else 0.0,
                    "min": round(float(non_null_df.min()), 4),
                    "max": round(float(non_null_df.max()), 4)
                }
            else:
                numeric_summary[col] = {
                    "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
                }

        categorical_summary = {}
        for col in categorical_cols:
            top_v = df[col].value_counts().head(5).to_dict()
            categorical_summary[col] = [{"value": str(val), "count": int(count)} for val, count in top_v.items()]

        statistical_summary = {
            "numeric": numeric_summary,
            "categorical": categorical_summary
        }

        # 4. PATTERNS & INSIGHTS
        # Auto-detect target column
        target_candidates = ["target", "label", "class", "churn", "default", "status", "y", "sold", "purchased", "admitted"]
        detected_target = None
        for col in df.columns:
            if col.lower() in target_candidates:
                detected_target = col
                break
        if not detected_target and len(df.columns) > 1:
            last_col = df.columns[-1]
            if not (last_col.lower().endswith("id") or df[last_col].nunique() == rows):
                detected_target = last_col

        # Find correlations above 0.7
        strong_correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            columns_list = list(corr_matrix.columns)
            for i in range(len(columns_list)):
                for j in range(i + 1, len(columns_list)):
                    col1 = columns_list[i]
                    col2 = columns_list[j]
                    val = corr_matrix.loc[col1, col2]
                    if not pd.isna(val) and abs(val) >= 0.7:
                        strong_correlations.append({
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(float(val), 4)
                        })

        # Outlier detection using IQR (Interquartile Range)
        outliers_info = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if not col_data.empty:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = int(outliers_mask.sum())
                    pct = float(outlier_count / len(col_data) * 100) if len(col_data) > 0 else 0.0
                    outliers_info[col] = {
                        "count": outlier_count,
                        "percentage": round(pct, 2)
                    }
                else:
                    outliers_info[col] = {"count": 0, "percentage": 0.0}
            else:
                outliers_info[col] = {"count": 0, "percentage": 0.0}

        duplicate_rows = int(df.duplicated().sum())

        patterns_insights = {
            "detected_target": detected_target,
            "strong_correlations": strong_correlations,
            "outliers": outliers_info,
            "duplicate_rows": duplicate_rows
        }

        # 5. WARNINGS
        general_warnings = []

        # Imbalanced target check (>80%)
        if detected_target:
            counts = df[detected_target].value_counts(normalize=True)
            if len(counts) > 1:
                major_class_pct = counts.iloc[0]
                if major_class_pct > 0.8:
                    general_warnings.append(
                        f"Target imbalance: Target column '{detected_target}' is heavily imbalanced. "
                        f"Class '{counts.index[0]}' represents {major_class_pct*100:.2f}% of the data."
                    )

        # Columns with all unique values (Potential Unique ID Columns)
        if rows > 1:
            for col in df.columns:
                if df[col].nunique() == rows:
                    general_warnings.append(
                        f"ID candidate: Column '{col}' has 100% unique values and might be an identifier/key."
                    )

        # Constant columns (no variation)
        for col in df.columns:
            if df[col].nunique() <= 1:
                general_warnings.append(
                    f"Constant column: Column '{col}' has no variation (only one unique value)."
                )
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_std = df[col].std()
                if not pd.isna(col_std) and col_std == 0:
                    general_warnings.append(
                        f"Constant column: Numeric column '{col}' has a standard deviation of 0."
                    )

        # helper function to inject Inter Google Font into the output Plotly HTML file
        def _inject_google_font(file_path: str):
            if os.path.exists(file_path):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                    font_link = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
                    content = content.replace("<head>", f"<head>{font_link}")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    print(f"Error injecting Google Font to {file_path}: {e}")

        # Premium dark template styling to match Dashboard theme
        layout_update = {
            "template": "plotly_dark",
            "paper_bgcolor": "#111118", # Sleek background container matching app card theme
            "plot_bgcolor": "rgba(10, 10, 15, 0.5)", # Subtle dark interior background
            "font": {"color": "#f8fafc", "family": "Inter, sans-serif"},
            "title": {
                "font": {"color": "#f8fafc", "size": 16, "family": "Inter, sans-serif"},
                "pad": {"b": 12},
                "x": 0.05
            },
            "xaxis": {
                "gridcolor": "#2a2a3a",
                "linecolor": "#2a2a3a",
                "zerolinecolor": "#2a2a3a",
                "title": {"font": {"color": "#94a3b8", "size": 13}}
            },
            "yaxis": {
                "gridcolor": "#2a2a3a",
                "linecolor": "#2a2a3a",
                "zerolinecolor": "#2a2a3a",
                "title": {"font": {"color": "#94a3b8", "size": 13}}
            },
            "coloraxis": {
                "colorbar": {
                    "thickness": 16,
                    "title": {"font": {"color": "#94a3b8", "size": 11}}
                }
            },
            "margin": {"l": 100, "r": 40, "t": 75, "b": 80} # Increased margins to prevent label truncation
        }

        # Correlation Heatmap
        heatmap_file = ""
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=".2f", # Formatted decimals for clean readability
                color_continuous_scale=[[0.0, "#ef4444"], [0.5, "#111118"], [1.0, "#10b981"]], # Modern red-to-dark-to-green diverging scale
                title="Correlation Matrix Heatmap",
                zmin=-1.0, # Center the scale at exactly 0.0 correlation
                zmax=1.0
            )
            fig_heatmap.update_layout(**layout_update)
            fig_heatmap.update_xaxes(showgrid=False)
            fig_heatmap.update_yaxes(showgrid=False)
            heatmap_file = os.path.join(OUTPUTS_DIR, "eda_correlation_heatmap.html")
            fig_heatmap.write_html(heatmap_file, include_plotlyjs="cdn")
            _inject_google_font(heatmap_file)

        # Distribution plots for numeric columns (limit to top 10 columns)
        dist_files = {}
        for col in numeric_cols[:10]:
            fig_dist = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                marginal="box",
                color_discrete_sequence=["#6366f1"] # Indigo primary accent
            )
            fig_dist.update_traces(marker_line_color='#111118', marker_line_width=1.5, opacity=0.85)
            fig_dist.update_layout(bargap=0.08, **layout_update)
            dist_file = os.path.join(OUTPUTS_DIR, f"eda_dist_{col}.html")
            fig_dist.write_html(dist_file, include_plotlyjs="cdn")
            _inject_google_font(dist_file)
            dist_files[col] = dist_file

        # Distribution plots for valid categorical columns (exclude IDs and constant columns, limit to top 5)
        valid_categorical_cols = [
            col for col in categorical_cols
            if df[col].nunique() > 1 and df[col].nunique() < rows
        ]
        for col in valid_categorical_cols[:5]:
            vc = df[col].value_counts().head(10).reset_index()
            vc.columns = [col, "Count"]
            fig_dist = px.bar(
                vc,
                x=col,
                y="Count",
                title=f"Distribution of {col} (Top 10)",
                color_discrete_sequence=["#8b5cf6"] # Purple secondary accent
            )
            fig_dist.update_traces(marker_line_color='#111118', marker_line_width=1.5, opacity=0.85)
            fig_dist.update_layout(bargap=0.2, **layout_update)
            dist_file = os.path.join(OUTPUTS_DIR, f"eda_dist_{col}.html")
            fig_dist.write_html(dist_file, include_plotlyjs="cdn")
            _inject_google_font(dist_file)
            dist_files[col] = dist_file

        visualizations = {
            "heatmap": heatmap_file,
            "distributions": dist_files
        }

        return {
            "basic_info": basic_info,
            "missing_values": missing_values,
            "statistical_summary": statistical_summary,
            "patterns_insights": patterns_insights,
            "warnings": general_warnings,
            "visualizations": visualizations
        }

    except Exception as e:
        raise RuntimeError(f"Error executing complete EDA analysis: {e!s}")


def read_dataset_info(file_path: str) -> str:
    """Reads a CSV dataset and returns a markdown summary of its schema and statistical description.

    This function wraps the comprehensive `run_eda()` tool and formats its output into a human-readable
    markdown summary for compatibility with existing agents.

    Args:
        file_path: The absolute or relative path to the CSV file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        analysis = run_eda(file_path)

        info = analysis["basic_info"]
        missing = analysis["missing_values"]
        stats = analysis["statistical_summary"]
        patterns = analysis["patterns_insights"]
        warnings = analysis["warnings"]

        lines = [
            f"# Dataset Summary: {os.path.basename(file_path)}",
            f"- **Rows**: {info['rows']}",
            f"- **Columns**: {info['columns']}",
            f"- **File Size**: {info['file_size_mb']} MB",
            f"- **Detected Target**: {patterns['detected_target'] or 'None'}",
            f"- **Duplicate Rows**: {patterns['duplicate_rows']}",
            ""
        ]

        if warnings:
            lines.append("## ⚠ Security & Schema Warnings")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("## Column Schema Details")
        lines.append("| Column Name | Data Type | Missing Count | Missing % | Status |")
        lines.append("| --- | --- | --- | --- | --- |")
        for col, dtype in info["column_types"].items():
            cnt = missing["counts"][col]
            pct = missing["percentages"][col]
            status = "⚠ >30% missing" if pct > 30 else "OK"
            lines.append(f"| {col} | {dtype} | {cnt} | {pct}% | {status} |")
        lines.append("")

        # Numeric Stats
        if stats["numeric"]:
            lines.append("## Numeric Columns Summary")
            lines.append("| Column Name | Mean | Median | Std Dev | Min | Max | Outliers (IQR) |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for col, n_stats in stats["numeric"].items():
                outlier = patterns["outliers"].get(col, {"count": 0, "percentage": 0.0})
                lines.append(
                    f"| {col} | {n_stats['mean']} | {n_stats['median']} | {n_stats['std']} | "
                    f"{n_stats['min']} | {n_stats['max']} | {outlier['count']} ({outlier['percentage']}%) |"
                )
            lines.append("")

        # Categorical Stats
        if stats["categorical"]:
            lines.append("## Categorical Columns (Top 5 Value Counts)")
            for col, vals in stats["categorical"].items():
                lines.append(f"### {col}")
                lines.append("| Value | Count |")
                lines.append("| --- | --- |")
                for item in vals:
                    lines.append(f"| {item['value']} | {item['count']} |")
                lines.append("")

        # Strong Correlations
        if patterns["strong_correlations"]:
            lines.append("## Strong Correlations (>= 0.7)")
            lines.append("| Feature 1 | Feature 2 | Correlation |")
            lines.append("| --- | --- | --- |")
            for corr in patterns["strong_correlations"]:
                lines.append(f"| {corr['col1']} | {corr['col2']} | {corr['correlation']} |")
            lines.append("")

        # Chart Links
        vis = analysis["visualizations"]
        if vis["heatmap"] or vis["distributions"]:
            lines.append("## Generated Visualizations (Plotly HTML)")
            if vis["heatmap"]:
                lines.append(f"- **Correlation Heatmap Matrix**: [Heatmap Chart](file://{vis['heatmap']})")
            for col, path in vis["distributions"].items():
                lines.append(f"- **{col} Distribution Plot**: [Distribution Chart](file://{path})")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error reading dataset: {e!s}"
