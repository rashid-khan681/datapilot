import logging
import os
import re

import requests
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datapilot.report_agent")

# Determine output directories
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(TOOLS_DIR)
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

class ReportTaskInput(BaseModel):
    file_path: str = Field(description="The absolute or relative path to the CSV file analyzed")
    goal: str = Field(description="The user's analytical or machine learning goal")
    eda_summary: str = Field(description="Summary from the EDA agent analysis stage")
    ml_summary: str = Field(description="Summary from the Machine Learning training stage")
    security_summary: str = Field(description="Summary from the Security audit review stage")
    domain: str | None = Field(default=None, description="The detected domain of the dataset")

class ReportAgentOutput(BaseModel):
    report_markdown: str = Field(description="The complete markdown source of the generated report")
    report_html: str = Field(description="The HTML rendered source of the generated report")
    report_file_path: str = Field(description="Absolute file path where the report.md file is saved")
    status: str = Field(default="complete", description="Completion status of the report stage")

def send_status(message: str, status: str = "running"):
    """Helper to send progress status updates to the FastAPI MCP server for WebSocket broadcast."""
    url = "http://localhost:8000/status/broadcast"
    try:
        requests.post(url, json={
            "agent": "DataPilot_Report_Agent",
            "message": message,
            "status": status
        }, timeout=2)
    except Exception as e:
        logger.warning(f"Failed to broadcast status update: {e}")

def parse_md_table(rows: list) -> str:
    """Helper to parse a list of markdown table rows and return a formatted HTML table."""
    html_table = ["<table>"]
    for idx, row in enumerate(rows):
        cells = [c.strip() for c in row.split('|')[1:-1]]
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
        html_table.append("  <tr>")
        tag = "th" if idx == 0 else "td"
        for cell in cells:
            html_table.append(f"    <{tag}>{cell}</{tag}>")
        html_table.append("  </tr>")
    html_table.append("</table>")
    return "\n".join(html_table)

def convert_markdown_to_premium_html(md: str) -> str:
    """Robust parser converting basic Markdown syntax into beautiful responsive CSS styled HTML."""
    html = md

    # 1. Parse tables
    lines = html.splitlines()
    new_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_rows.append(stripped)
        else:
            if in_table:
                formatted_table = parse_md_table(table_rows)
                new_lines.append(formatted_table)
                table_rows = []
                in_table = False
            new_lines.append(line)
    if in_table:
        formatted_table = parse_md_table(table_rows)
        new_lines.append(formatted_table)

    html = "\n".join(new_lines)

    # 2. Bolds and Italics
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

    # 3. Headers
    html = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    # 4. Bullet & Numbered lists + Blockquotes
    processed_lines = []
    in_list = False
    in_num_list = False

    for line in html.splitlines():
        bullet_match = re.match(r'^\s*[-*•]\s+(.+)$', line)
        num_match = re.match(r'^\s*(\d+)\.\s+(.+)$', line)
        blockquote_match = re.match(r'^>\s*(.+)$', line)

        if bullet_match:
            if in_num_list:
                processed_lines.append("</ol>")
                in_num_list = False
            if not in_list:
                processed_lines.append("<ul>")
                in_list = True
            processed_lines.append(f"<li>{bullet_match.group(1)}</li>")
        elif num_match:
            if in_list:
                processed_lines.append("</ul>")
                in_list = False
            if not in_num_list:
                processed_lines.append('<ol class="rec-list">')
                in_num_list = True
            processed_lines.append(f"<li>{num_match.group(2)}</li>")
        elif blockquote_match:
            if in_list:
                processed_lines.append("</ul>")
                in_list = False
            if in_num_list:
                processed_lines.append("</ol>")
                in_num_list = False
            processed_lines.append(f'<div class="bottom-line-box">{blockquote_match.group(1)}</div>')
        else:
            if in_list:
                processed_lines.append("</ul>")
                in_list = False
            if in_num_list:
                processed_lines.append("</ol>")
                in_num_list = False
            processed_lines.append(line)

    if in_list:
        processed_lines.append("</ul>")
    if in_num_list:
        processed_lines.append("</ol>")

    html = "\n".join(processed_lines)

    # 5. Paragraph wrapping
    final_lines = []
    for line in html.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        if (line_strip.startswith('<h') or line_strip.startswith('<ul') or
            line_strip.startswith('<ol') or line_strip.startswith('<li') or
            line_strip.startswith('<tr') or line_strip.startswith('<td') or
            line_strip.startswith('<th') or line_strip.startswith('<table') or
            line_strip.startswith('<div') or line_strip.startswith('</') or
            line_strip.startswith('<!')):
            final_lines.append(line)
        else:
            final_lines.append(f"<p>{line}</p>")

    html_body = "\n".join(final_lines)

    # Premium wrapper template
    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataPilot Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #090d16;
            color: #f3f4f6;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            background: linear-gradient(135deg, #8a2be2 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #00f2fe;
            font-size: 1.8rem;
            margin-top: 40px;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }}
        h3 {{
            color: #a24bfa;
            font-size: 1.3rem;
            margin-top: 25px;
        }}
        p {{
            margin-bottom: 15px;
            font-size: 1.05rem;
            color: #d1d5db;
        }}
        ul, ol {{
            margin-left: 20px;
            margin-bottom: 20px;
        }}
        li {{
            margin-bottom: 8px;
            color: #d1d5db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: rgba(17, 25, 40, 0.75);
            backdrop-filter: blur(16px);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        th {{
            background-color: rgba(138, 43, 226, 0.2);
            color: #f3f4f6;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        strong {{
            color: #f3f4f6;
            font-weight: 600;
        }}
        code {{
            font-family: 'JetBrains Mono', monospace;
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #a5f3fc;
        }}
        /* ── Recommendation Cards ── */
        ol.rec-list {{
            list-style: none;
            padding: 0;
            margin: 0;
            counter-reset: rec-counter;
        }}
        ol.rec-list li {{
            counter-increment: rec-counter;
            position: relative;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 12px;
            padding: 16px 20px 16px 14px;
            margin-bottom: 12px;
            color: #e2e8f0;
            font-size: 0.97rem;
            line-height: 1.6;
            transition: border-color 0.2s, background 0.2s;
        }}
        ol.rec-list li:hover {{
            border-color: rgba(99,102,241,0.55);
            background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(139,92,246,0.10) 100%);
        }}
        ol.rec-list li::before {{
            content: counter(rec-counter);
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 50%;
            font-size: 0.8rem;
            font-weight: 700;
            color: #fff;
            margin-top: 1px;
        }}
        /* ── Bottom Line Callout ── */
        .bottom-line-box {{
            background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.12) 100%);
            border: 1.5px solid rgba(99,102,241,0.45);
            border-left: 5px solid #6366f1;
            border-radius: 12px;
            padding: 18px 22px;
            margin: 24px 0 8px;
            font-size: 1.02rem;
            color: #c7d2fe;
            line-height: 1.7;
        }}
        .bottom-line-box strong {{
            color: #a5b4fc;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""
    return styled_html

def save_report_both(markdown_content: str) -> dict:
    """Saves the report as markdown and compiles it to a styled premium HTML page.

    Args:
        markdown_content: The full markdown report content.
    """
    send_status("Writing your report...", "running")

    # Save report.md
    md_path = os.path.join(OUTPUTS_DIR, "report.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    except Exception as e:
        logger.error(f"Failed to save report.md: {e}")

    # Convert markdown to HTML
    html_content = convert_markdown_to_premium_html(markdown_content)

    # Save report.html
    html_path = os.path.join(OUTPUTS_DIR, "report.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Failed to save report.html: {e}")

    send_status("Report Ready!", "complete")

    return {
        "report_markdown": markdown_content,
        "report_html": html_content,
        "report_file_path": md_path
    }

def get_model():
    return Gemini(
        model="gemini-1.5-flash",
        retry_options=types.HttpRetryOptions(attempts=1),
    )

report_agent = Agent(
    name="DataPilot_Report_Agent",
    model=get_model(),
    tools=[save_report_both],
    input_schema=ReportTaskInput,
    output_schema=ReportAgentOutput,
    description="Business Intelligence Report Generator. Compiles all multi-agent execution summaries, metrics, and warnings into a final report.",
    instruction="""You are a Business Intelligence Reporter working inside the DataPilot platform.
Your task is to compile a stunning, comprehensive, and professional Data Science Report using the inputs from the EDA, ML, and Security agents.

Your workflow:
1. Gather results from input fields:
   - 'eda_summary'
   - 'ml_summary'
   - 'security_summary'
   - 'domain'
2. Generate a beautifully formatted Markdown report containing EXACTLY these sections (use emojis and write in simple English, readable for both technical and non-technical stakeholders):

   # DataPilot Analysis Report
   ## Executive Summary (Must be 3 lines max)
   ## What We Found In Your Data
   ## Machine Learning Results
   ## Key Business Insights
   ## Recommended Actions (Include a numbered list)
   ## Security Summary
   ## Files Generated

3. For the 'Machine Learning Results' section, you MUST include this block EXACTLY ONCE and nowhere else:

   RULE — TOP RISK FACTORS FORMAT (include ONCE only, here in Machine Learning Results):
   Format it exactly like:
   🔍 Top Risk Factors Identified:
    1. [feature_1]      → [X]% importance
    2. [feature_2]      → [Y]% importance
    3. [feature_3]      → [Z]% importance
   (continue for all features found in ml_summary)
   Ensure the arrow → aligns perfectly. Do NOT repeat this list in any other section.

4. For the 'Key Business Insights' section, you MUST produce EXACTLY 3 real data-driven insights:

   RULE — NO PLACEHOLDER TEXT: Never write generic text like "Optimizations and automated classification
   pipelines are active." or any filler that adds zero factual value.

   Use this EXACT format:
   **INSIGHT 1 — DATASET OVERVIEW:** Your dataset of [X] [patients/customers/employees] shows a [Y]%
   rate of [target_column]. This means roughly [Z] out of every 10 [people] in your data are at risk.

   **INSIGHT 2 — STRONGEST PATTERN:** The single strongest pattern found is `[top_feature]`.
   [Patients/Customers] with [feature_value] are [X]% more likely to have [target_column] than those without it.
   Use the actual averages from the GROUNDED MACHINE LEARNING STATS provided in ml_summary.

   **INSIGHT 3 — GOOD NEWS:** On the positive side, [X]% of [patients/customers] in your data show
   low risk profiles — meaning DataPilot can help you focus resources on the [Y]% who need attention most.

   Always use actual numbers from the dataset. Use domain-appropriate terms (patients/customers/employees/students).

5. For the 'Recommended Actions' section, strictly follow these rules:

   RULE 1 — ONLY REAL DATA:
   Every statistic and feature name must come directly from the actual data (from ml_summary GROUNDED stats).
   Never hallucinate statistics.

   RULE 2 — NO DUPLICATE FEATURES (Critical Fix):
   - Generate EXACTLY 5 recommendations.
   - Each recommendation must be based on ONE UNIQUE feature — never repeat the same feature twice.
   - Step 1: List ALL unique important features from the ML results.
   - Step 2: Generate one recommendation per unique feature.
   - Step 3: If fewer than 5 unique features exist, fill remaining slots with:
     Slot A — COMBINATION RISK: "[Term] with BOTH high-risk [feature1] AND high-risk [feature2]
     simultaneously show the highest overall risk — prioritize this combined segment."
     Slot B — PREVENTIVE ACTION: "Regular monitoring of [top_feature] every 3 months is recommended
     for all [term] as it is the strongest predictor of [target_column]."
     Slot C — LOW RISK INSIGHT: "[Term] without high-risk [top_feature] values show the lowest risk
     profile — use them as your healthy baseline when calibrating risk thresholds."

   RULE 3 — FORMAT:
   "[Patients/Customers/Employees/Students] where [feature_name] is [ABOVE/BELOW] [threshold] show
   [X]% higher risk of [target_column] — [Action to take]"
   (e.g., "Patients where cholesterol_total is ABOVE 260 show 78% higher risk of heart_disease
   — schedule immediate lipid panel review for these patients.")

   RULE 4 — REAL NUMBERS ONLY: Use exact averages/percentages from ml_summary. Never invent numbers.

   RULE 5 — ACTUAL COLUMN NAMES: Always use exact column names from the CSV.

   RULE 6 — BOTTOM LINE: Must include actual model name, actual accuracy, and actual target column.
   Format EXACTLY: "Bottom Line: [model_name] predicts [actual_target_column] with [actual_accuracy]%
   accuracy. The single most important action is to monitor [top_feature] as it is the strongest
   predictor in your dataset."

   DOMAIN-SPECIFIC TONE & STYLE (Adjust based on the 'domain' field):
   - MEDICAL: Clinical, doctor-friendly. Serious, precise, life-critical. Term is Patients.
   - FINANCE: Risk-focused, regulatory. Conservative, data-driven. Term is Customers.
   - CUSTOMER: Business, marketing-friendly. Growth-focused, actionable. Term is Customers.
   - HR: People-management focused. Empathetic, policy-oriented. Term is Employees.
   - EDUCATION: Academic focused. Supportive, improvement-oriented. Term is Students.
   - GENERAL (or other): Business neutral. Professional, clear. Term is Customers.

6. Be specific with numbers, percentages, and metrics (F1-score, Accuracy, etc.).
7. Call the 'save_report_both' tool, passing the generated Markdown string.
8. Populate the output schema fields:
   - 'report_markdown': The generated Markdown string.
   - 'report_html': The HTML output returned by the 'save_report_both' tool.
   - 'report_file_path': The path where the markdown report was saved.
   - 'status': Always set to "complete".
"""
)
