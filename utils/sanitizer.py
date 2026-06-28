import csv
import os
import re


def sanitize_goal(goal: str) -> str:
    """
    Removes potentially dangerous content from user text input (no SQL, no script tags, no shell commands).
    Strips extra whitespace from user goal text.
    Returns clean, safe inputs only.
    Raises ValueError with a helpful message on failure.
    """
    if not goal or not isinstance(goal, str):
        raise ValueError("Goal must be a non-empty string.")

    # Strip leading/trailing whitespace
    clean_goal = goal.strip()

    # Check/remove script tags
    script_pattern = re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', re.IGNORECASE)
    if script_pattern.search(clean_goal):
        clean_goal = script_pattern.sub('', clean_goal)

    # Strip any remaining HTML/XML tags
    html_pattern = re.compile(r'<[^>]+>')
    if html_pattern.search(clean_goal):
        clean_goal = html_pattern.sub('', clean_goal)

    # SQL injection patterns
    sql_patterns = [
        re.compile(r'\bunion\b.*\bselect\b', re.IGNORECASE),
        re.compile(r'\bselect\b.*\bfrom\b', re.IGNORECASE),
        re.compile(r'\binsert\b.*\binto\b', re.IGNORECASE),
        re.compile(r'\bdrop\b.*\btable\b', re.IGNORECASE),
        re.compile(r'\bdelete\b.*\bfrom\b', re.IGNORECASE),
        re.compile(r'\bupdate\b.*\bset\b', re.IGNORECASE),
    ]

    for pattern in sql_patterns:
        if pattern.search(clean_goal):
            clean_goal = pattern.sub('', clean_goal)

    # Shell commands/metacharacters check
    shell_chars = [';', '|', '&', '`', '$(', ')', '>', '<']
    for char in shell_chars:
        clean_goal = clean_goal.replace(char, '')

    # Strip any extra whitespace left after replacements
    clean_goal = re.sub(r'\s+', ' ', clean_goal).strip()

    if not clean_goal:
        raise ValueError("Goal text became empty or invalid after security sanitization.")

    return clean_goal


def sanitize_csv_file(file_path: str) -> str:
    """
    Validates uploaded file is actually a CSV:
    - Check extension is .csv
    - Check file size is under 50MB
    - Check file content is valid CSV (not binary, parses successfully)
    Raises ValueError with a helpful message on failure.
    """
    if not file_path:
        raise ValueError("File path must be provided.")

    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Check extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.csv':
        raise ValueError(f"Invalid file extension '{ext}'. Only .csv files are supported.")

    # Check file size (50MB = 50 * 1024 * 1024 bytes)
    max_bytes = 50 * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size > max_bytes:
        raise ValueError(
            f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds the maximum limit of 50MB."
        )

    # Check content (not binary, parses as CSV)
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            # Read first 10KB to sample
            sample = f.read(10240)
            if not sample:
                raise ValueError("The uploaded CSV file is empty.")

            # Check for null bytes indicating binary
            if '\x00' in sample:
                raise ValueError("Binary content detected. The file is not a valid CSV.")

            # Try to sniff the CSV dialect to confirm format
            try:
                dialect = csv.Sniffer().sniff(sample)
                if not dialect.delimiter:
                    raise ValueError("Unable to identify CSV delimiter.")
            except Exception:
                # If sniffer fails, check if the first line contains at least one common delimiter
                first_line = sample.splitlines()[0] if sample.splitlines() else ""
                delimiters = [',', ';', '\t', '|']
                if not any(d in first_line for d in delimiters):
                    raise ValueError("The file content is not structured as delimited CSV data.")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to validate CSV content: {e!s}")

    return file_path


def validate_safe_path(path: str) -> str:
    """
    Ensures the path is resolved and lies within the permitted workspace directory
    or system temp directories. Prevents path traversal attacks.
    """
    if not path:
        return path

    # Resolve symlinks and normalize the path
    abs_path = os.path.realpath(os.path.abspath(path))

    # Permitted roots: WORKSPACE_ROOT
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(utils_dir)
    allowed_root = os.path.realpath(os.path.abspath(workspace_root))

    # Check against allowed root and standard temp upload folders
    allowed_paths = [allowed_root, "/private/var", "/var/tmp", "/tmp"]

    is_safe = False
    for root in allowed_paths:
        resolved_root = os.path.realpath(os.path.abspath(root))
        # Use os.sep-appended comparison to prevent prefix bypass
        # e.g., "/allowed_root_extra" should NOT match "/allowed_root"
        if abs_path == resolved_root or abs_path.startswith(resolved_root + os.sep):
            is_safe = True
            break

    if not is_safe:
        raise ValueError("Access Denied: Path is outside permitted directories.")

    return abs_path


