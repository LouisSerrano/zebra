"""Download datasets from Hugging Face.

Usage:
    python download_data_hugging_face.py --datasets vorticity wave gs
    python download_data_hugging_face.py --datasets all
    python download_data_hugging_face.py --datasets vorticity --data_dir /my/custom/path
"""

import argparse
import os

from huggingface_hub import snapshot_download

DATASETS = {
    # 2D datasets (from Armand Kassai Koupaï)
    "vorticity": "sogeeking/vorticity",
    "vorticity_ood": "sogeeking/vorticity_ood",
    "wave": "sogeeking/wave",
    "wave_ood": "sogeeking/wave_ood",
    "gs": "sogeeking/gs",
    "gs_ood": "sogeeking/gs_ood",
    # 1D datasets
    "combined_equation": "sogeeking/combined-equation-2",
    "advection_diffusion": "sogeeking/advection-diffusion",
    "heat_nu_forcing2": "sogeeking/heat-nu-forcing-2",
    "burgers_nu_forcing2": "sogeeking/burgers-nu-forcing-2",
}


def download_dataset(name: str, repo_id: str, data_dir: str):
    local_dir = os.path.join(data_dir, name)
    print(f"Downloading {name} from {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print(f"  Done: {name}\n")


def main():
    parser = argparse.ArgumentParser(description="Download Zebra/ENMA datasets from Hugging Face.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=list(DATASETS.keys()) + ["all"],
        help="Datasets to download. Use 'all' to download everything.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory where datasets will be saved (default: ./data).",
    )
    args = parser.parse_args()

    if "all" in args.datasets:
        to_download = DATASETS
    else:
        to_download = {k: DATASETS[k] for k in args.datasets}

    os.makedirs(args.data_dir, exist_ok=True)

    print(f"Downloading {len(to_download)} dataset(s) to {os.path.abspath(args.data_dir)}\n")
    for name, repo_id in to_download.items():
        download_dataset(name, repo_id, args.data_dir)

    print("All downloads complete.")


if __name__ == "__main__":
    main()
