from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


COMPETITION = "house-prices-advanced-regression-techniques"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ZIP_FILE = DATA_DIR / f"{COMPETITION}.zip"


def run_command(command: list[str]) -> None:
    """Run a command and stop with a useful message if it fails."""
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(
            "The Kaggle CLI was not found.\n"
            "Install it with: python -m pip install kaggle"
        )
        sys.exit(1)
    except subprocess.CalledProcessError:
        print(
            "\nKaggle download failed.\n"
            "Check that you authenticated and accepted the competition rules."
        )
        sys.exit(1)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading the Kaggle House Prices dataset...")

    run_command(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DATA_DIR),
            "--force",
        ]
    )

    if not ZIP_FILE.exists():
        print(f"Expected ZIP file was not found: {ZIP_FILE}")
        sys.exit(1)

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        archive.extractall(DATA_DIR)

    ZIP_FILE.unlink()

    required_file = DATA_DIR / "train.csv"

    if not required_file.exists():
        print("Download completed, but train.csv was not found.")
        sys.exit(1)

    print(f"Dataset ready: {required_file}")

    # test.csv and sample_submission.csv are unnecessary for this demo.
    for optional_file in ["test.csv", "sample_submission.csv"]:
        path = DATA_DIR / optional_file
        if path.exists():
            path.unlink()

    description = DATA_DIR / "data_description.txt"
    if description.exists():
        shutil.move(description, PROJECT_DIR / "data_description.txt")


if __name__ == "__main__":
    main()
