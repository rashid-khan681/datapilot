import os
import shutil

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
UPLOADS_DIR = os.path.join(WORKSPACE_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

def download_kaggle_dataset(dataset_slug: str) -> dict:
    """Downloads a dataset from Kaggle and saves it to the uploads directory.
    
    Requires Kaggle credentials to be configured. If credentials are missing,
    it falls back to using the pre-generated customer churn dataset for demo purposes.
    
    Args:
        dataset_slug: The Kaggle dataset identifier in format 'username/dataset-name'.
    """
    # Safeguard check
    has_creds = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

    if not has_creds:
        # Fallback to local demo dataset if available
        demo_src = os.path.join(UPLOADS_DIR, "customer_churn.csv")
        dest_filename = f"{dataset_slug.replace('/', '_')}.csv"
        demo_dest = os.path.join(UPLOADS_DIR, dest_filename)

        if os.path.exists(demo_src):
            shutil.copy(demo_src, demo_dest)
            return {
                "status": "success",
                "message": f"Kaggle credentials not found. Fell back to pre-generated customer churn demo dataset, saved as '{dest_filename}'",
                "csv_files": [dest_filename],
                "csv_paths": [demo_dest],
                "primary_csv": demo_dest
            }
        else:
            return {
                "status": "error",
                "message": "Kaggle credentials not configured, and demo 'customer_churn.csv' not found. Please set KAGGLE_USERNAME and KAGGLE_KEY env vars."
            }

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print(f"Downloading Kaggle dataset '{dataset_slug}' to '{UPLOADS_DIR}'...")
        api.dataset_download_files(dataset_slug, path=UPLOADS_DIR, unzip=True)

        # Find downloaded CSV files
        downloaded_files = os.listdir(UPLOADS_DIR)
        csv_files = [f for f in downloaded_files if f.endswith('.csv')]

        if not csv_files:
            return {
                "status": "warning",
                "message": f"Dataset downloaded but no CSV files found. Available files: {downloaded_files}",
                "files": [os.path.join(UPLOADS_DIR, f) for f in downloaded_files]
            }

        csv_paths = [os.path.join(UPLOADS_DIR, f) for f in csv_files]

        return {
            "status": "success",
            "message": f"Successfully downloaded and extracted Kaggle dataset: {dataset_slug}",
            "csv_files": csv_files,
            "csv_paths": csv_paths,
            "primary_csv": csv_paths[0]
        }

    except Exception as e:
        # Fallback to local demo if API call fails
        demo_src = os.path.join(UPLOADS_DIR, "customer_churn.csv")
        dest_filename = f"{dataset_slug.replace('/', '_')}.csv"
        demo_dest = os.path.join(UPLOADS_DIR, dest_filename)

        if os.path.exists(demo_src):
            shutil.copy(demo_src, demo_dest)
            return {
                "status": "success",
                "message": f"Kaggle API call failed ({e!s}). Fell back to pre-generated customer churn demo dataset, saved as '{dest_filename}'",
                "csv_files": [dest_filename],
                "csv_paths": [demo_dest],
                "primary_csv": demo_dest
            }
        return {
            "status": "error",
            "message": f"Kaggle download failed: {e!s}"
        }

def search_kaggle_datasets(query: str) -> list:
    """Searches Kaggle for datasets matching the query.
    
    If credentials are not configured, returns a pre-curated list of matching datasets.
    
    Args:
        query: The search query string.
    """
    has_creds = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

    # Pre-curated mock datasets for demo purposes
    mock_datasets = [
        {
            "ref": "blastchar/telco-customer-churn",
            "title": "Telco Customer Churn",
            "size": "172 KB",
            "description": "Predict behavior to retain customers. Favorable for binary classification.",
            "download_count": "540k downloads"
        },
        {
            "ref": "heptalophos/titanic",
            "title": "Titanic Dataset",
            "size": "60 KB",
            "description": "Complete survival analysis dataset. Great for binary classification prototyping.",
            "download_count": "120k downloads"
        },
        {
            "ref": "yasserh/housing-prices-dataset",
            "title": "Housing Prices Dataset",
            "size": "20 KB",
            "description": "Dataset containing housing pricing information for regression analysis.",
            "download_count": "45k downloads"
        },
        {
            "ref": "uciml/iris",
            "title": "Iris Species",
            "size": "4 KB",
            "description": "Classify iris species. Classic multi-class classification dataset.",
            "download_count": "320k downloads"
        }
    ]

    if not has_creds:
        # Filter mock datasets based on query
        q = query.lower()
        results = [d for d in mock_datasets if q in d["title"].lower() or q in d["description"].lower() or q in d["ref"].lower()]
        if not results:
            # If no matches, return the full list as suggestions
            return mock_datasets
        return results

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        datasets = api.dataset_list(search=query)

        results = []
        for d in datasets[:10]: # limit to 10 results
            results.append({
                "ref": d.ref,
                "title": d.title,
                "size": d.size,
                "description": getattr(d, 'description', 'No description available.'),
                "download_count": f"{d.downloadCount} downloads"
            })

        if not results:
            return [{"ref": "None", "title": "No datasets found.", "size": "", "description": "Try searching another term.", "download_count": ""}]
        return results

    except Exception:
        # Return fallback mock list if API call fails
        return [d for d in mock_datasets if query.lower() in d["title"].lower() or query.lower() in d["description"].lower()]
