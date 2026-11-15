"""Google Colab bootstrap.

Notebook usage (first cells of every notebook):

    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.append('/content/drive/MyDrive/Coursera/projects_github/nyc_taxi_2023_v2')

    from src.colab_setup import setup_environment
    setup_environment()

The ``sys.path`` line is what lets Python find ``src`` in the first place; after
that ``setup_environment`` moves into the project folder and prepares the
session. Outside Colab it only locates the repository root.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

__all__ = ["setup_environment", "PROJECT_PATHS_TO_TRY", "in_colab"]

PROJECT_PATHS_TO_TRY = [
    "/content/drive/MyDrive/Coursera/projects_github/nyc_taxi_2023_v2",
]


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def setup_environment(paths_to_try: list[str] | None = None) -> Path:
    """Mount Drive if needed, cd into the project folder, prepare the session.

    Returns the project root.
    """
    paths_to_try = paths_to_try or PROJECT_PATHS_TO_TRY

    # Mount Drive only if the notebook has not already done so.
    if in_colab() and not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")

    folder_found = False
    for target_path in paths_to_try:
        if os.path.exists(target_path):
            os.chdir(target_path)                       # same as: %cd "$target_path"
            print(f"\nSuccessfully changed directory to: {os.getcwd()}")
            folder_found = True
            break

    if not folder_found and in_colab():
        print("\nWarning: Could not find the project folder. "
              "Please verify your folder name in Google Drive:")
        for p in paths_to_try:
            print(f" - {p}")

    root = Path.cwd()
    if not (root / "src").exists() and (root.parent / "src").exists():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Same session defaults as the original notebook
    warnings.filterwarnings("ignore")
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    for sub in ("data/raw", "data/processed"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    print("Environment ready. Project root:", root)
    return root
