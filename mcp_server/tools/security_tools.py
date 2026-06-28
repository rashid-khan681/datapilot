import datetime
import os
import re

import numpy as np
import pandas as pd

# Determine directories
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def scan_code_security(code: str) -> dict:
    """Scans the Python code for security issues and machine learning best practices/methodology errors.

    This function performs static code scanning on code strings for dangerous functions (eval, exec)
    or potential target leakage.

    Args:
        code: The Python code to analyze.
    """
    warnings = []

    # 1. Unsafe calls
    if re.search(r'\b(exec|eval)\b', code):
        warnings.append({
            "severity": "CRITICAL",
            "category": "Security",
            "message": "Use of 'eval' or 'exec' detected. This is a severe security vulnerability."
        })

    if re.search(r'\b(subprocess|os\.system|shutil|sys)\b', code):
        warnings.append({
            "severity": "WARNING",
            "category": "Security",
            "message": "System commands or subprocesses detected. Unsafe code execution check required."
        })

    # 2. Data Leakage (Target variable used in training features)
    if "fit_transform" in code or "fit" in code:
        if "train_test_split" not in code:
            warnings.append({
                "severity": "HIGH",
                "category": "ML Methodology",
                "message": "Model training detected without 'train_test_split'. Ensure you evaluate the model on unseen data."
            })

        if "fit" in code and "train_test_split" in code:
            if code.find("fit") < code.find("train_test_split") and ("StandardScaler" in code or "MinMaxScaler" in code or "OneHotEncoder" in code):
                warnings.append({
                    "severity": "HIGH",
                    "category": "ML Methodology",
                    "message": "Data preprocessing fit (e.g. StandardScaler.fit) appears to happen before train_test_split. This causes data leakage."
                })

    # 3. Model validation check
    if "fit(" in code:
        if "predict" not in code and "predict_proba" not in code:
            warnings.append({
                "severity": "MEDIUM",
                "category": "ML Methodology",
                "message": "Model fit was called, but no prediction/evaluation was done. Ensure you measure accuracy/ROC-AUC."
            })

    status = "PASS"
    if any(w["severity"] == "CRITICAL" for w in warnings):
        status = "FAIL"
    elif any(w["severity"] in ["HIGH", "WARNING"] for w in warnings):
        status = "WARNING"

    return {
        "status": status,
        "warnings": warnings,
        "total_warnings": len(warnings)
    }


def save_report(content: str) -> str:
    """Saves the final markdown report to the outputs directory.

    Args:
        content: The Markdown content of the report.
    """
    report_path = os.path.join(OUTPUTS_DIR, "report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved report to: {report_path}"
    except Exception as e:
        return f"Error saving report: {e!s}"


def review_code(file_path: str, target: str | None = None, code: str | None = None) -> dict:
    """
    Performs a comprehensive data privacy and security review on a dataset.

    Checks include:
      1. DATA PRIVACY SCAN: Regex scans column names and samples for emails, phones, SSNs, credit cards
      2. DATA LEAKAGE CHECK: Checks for direct target overlap or extreme correlations (>0.99)
      3. MODEL FAIRNESS CHECK: Flags sensitive demographic attributes (gender, race, age, religion)
      4. INPUT VALIDATION: Checks CSV integrity and SQL/HTML injection attempts in column headers
      5. OUTPUT SECURITY: Verifies outputs directory safety and scans reports for leaked PII
      6. AUDIT LOGGING: Saves full audit report to outputs/security_audit.txt

    Args:
        file_path: Path to the CSV dataset.
        target: Target column name (optional).
        code: Code or report string to scan (optional).

    Returns:
        Structured dictionary with issue lists, fix suggestions, score, and safety flag.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found at: {file_path}")

    issues = []
    start_time = datetime.datetime.now()

    try:
        # 1. INPUT VALIDATION (Integrity Check)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            issues.append({
                "severity": "CRITICAL",
                "category": "Input",
                "message": f"CSV dataset is corrupted or failed to parse: {e!s}",
                "suggestion": "Re-export the dataset into a valid standard comma-separated CSV file."
            })
            # Return immediately if file cannot be read
            return {
                "issues": issues,
                "score": 0,
                "safe_to_proceed": False
            }

        # SQL/HTML Injection check in column names
        injection_chars = [';', '--', '/*', "'", '"', '<', '>', '|', '$', '\\']
        for col in df.columns:
            if any(char in col for char in injection_chars):
                issues.append({
                    "severity": "CRITICAL",
                    "category": "Input",
                    "message": f"Potential injection characters detected in column name '{col}'.",
                    "suggestion": f"Rename column '{col}' to contain alphanumeric characters only."
                })

        # 2. DATA PRIVACY SCAN (PII Detection)
        # Regexes for scanning sample rows
        email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        phone_regex = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b')
        ssn_regex = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        cc_regex = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

        for col in df.columns:
            col_lower = col.lower()

            # Check column name first
            is_pii_name = any(k in col_lower for k in ["email", "phone", "ssn", "socialsecurity", "creditcard", "ccnum", "passport", "password"])
            if is_pii_name:
                severity = "CRITICAL" if any(k in col_lower for k in ["ssn", "password", "creditcard", "passport"]) else "WARNING"
                issues.append({
                    "severity": severity,
                    "category": "Privacy",
                    "message": f"Column '{col}' matches a sensitive PII name pattern.",
                    "suggestion": f"Drop column '{col}' from features before training unless strictly required."
                })
                continue

            # Scan first 50 rows of content
            sample_vals = df[col].dropna().head(50).astype(str)
            has_email = any(email_regex.search(val) for val in sample_vals)
            has_ssn = any(ssn_regex.search(val) for val in sample_vals)
            has_cc = any(cc_regex.search(val) for val in sample_vals)
            has_phone = any(phone_regex.search(val) for val in sample_vals)

            if has_ssn or has_cc:
                issues.append({
                    "severity": "CRITICAL",
                    "category": "Privacy",
                    "message": f"Column '{col}' contains values matching SSN or Credit Card patterns.",
                    "suggestion": f"Drop column '{col}' immediately to protect sensitive personal data."
                })
            elif has_email or has_phone:
                issues.append({
                    "severity": "WARNING",
                    "category": "Privacy",
                    "message": f"Column '{col}' contains values matching Email or Phone number patterns.",
                    "suggestion": f"Remove or anonymize column '{col}' before using it in machine learning models."
                })

        # 3. DATA LEAKAGE CHECK
        if target:
            if target not in df.columns:
                issues.append({
                    "severity": "WARNING",
                    "category": "Leakage",
                    "message": f"Target column '{target}' was specified but not found in CSV.",
                    "suggestion": "Ensure the target column is spelled correctly."
                })
            else:
                # Factorize target if categorical to calculate correlation
                if not pd.api.types.is_numeric_dtype(df[target]):
                    target_series = pd.Series(pd.factorize(df[target])[0])
                else:
                    target_series = df[target]

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    if col == target:
                        continue
                    corr = df[col].corr(target_series)
                    if not pd.isna(corr) and abs(corr) > 0.99:
                        issues.append({
                            "severity": "CRITICAL",
                            "category": "Leakage",
                            "message": f"Column '{col}' has an extremely high correlation ({corr:.4f}) with target '{target}'.",
                            "suggestion": f"Remove column '{col}' from features as it represents data leakage (directly exposes target)."
                        })

        # 4. MODEL FAIRNESS CHECK
        sensitive_keywords = ["gender", "sex", "age", "religion", "race", "ethnicity", "nationality", "marital", "disability", "sexual"]
        for col in df.columns:
            if target and col == target:
                continue
            if any(k in col.lower() for k in sensitive_keywords):
                issues.append({
                    "severity": "WARNING",
                    "category": "Fairness",
                    "message": f"Column '{col}' represents a sensitive demographic attribute.",
                    "suggestion": f"Check if using '{col}' introduces demographic bias. Consider evaluating fairness metrics or dropping it."
                })

        # 5. OUTPUT SECURITY
        # safe directory checks
        is_safe = os.path.abspath(OUTPUTS_DIR).startswith(os.path.abspath(WORKSPACE_ROOT))
        if not is_safe:
            issues.append({
                "severity": "CRITICAL",
                "category": "Output",
                "message": f"Output directory {OUTPUTS_DIR} is outside the safe workspace root {WORKSPACE_ROOT}.",
                "suggestion": "Configure output directory to reside inside the project workspace directory."
            })

        # scan report/code string
        if code:
            if email_regex.search(code) or ssn_regex.search(code) or cc_regex.search(code):
                issues.append({
                    "severity": "WARNING",
                    "category": "Output",
                    "message": "Sensitive PII details detected inside the report/code content.",
                    "suggestion": "Anonymize or remove personal identifiers before exporting reports."
                })

        # Score calculation
        score = 100
        for issue in issues:
            if issue["severity"] == "CRITICAL":
                score -= 25
            elif issue["severity"] in ["HIGH", "WARNING"]:
                score -= 10
            else:
                score -= 2
        score = max(0, score)

        safe_to_proceed = not any(issue["severity"] == "CRITICAL" for issue in issues)

        # 6. AUDIT LOG WRITER
        audit_log_path = os.path.join(OUTPUTS_DIR, "security_audit.txt")
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(f"=== SECURITY AUDIT LOG {timestamp} ===\n")
            f.write(f"File Path: {file_path}\n")
            f.write(f"Target: {target}\n")
            f.write(f"Elapsed Time: {(datetime.datetime.now() - start_time).total_seconds():.4f}s\n")
            f.write(f"Issues Found: {len(issues)}\n")
            for idx, issue in enumerate(issues):
                f.write(f"  {idx+1}. [{issue['severity']}] ({issue['category']}) {issue['message']}\n")
                f.write(f"     Fix Suggestion: {issue['suggestion']}\n")
            f.write(f"Overall Security Score: {score}/100\n")
            f.write(f"Safe to Proceed: {safe_to_proceed}\n")
            f.write("="*60 + "\n\n")

        return {
            "issues": issues,
            "score": score,
            "safe_to_proceed": safe_to_proceed,
            "audit_log": audit_log_path
        }

    except Exception as e:
        raise RuntimeError(f"Security review failed: {e!s}")
