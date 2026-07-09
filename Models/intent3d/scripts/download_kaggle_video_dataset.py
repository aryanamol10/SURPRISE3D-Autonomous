"""Download and prepare a Kaggle video dataset for the raw-video pipeline."""

import argparse
import json
import os
import shutil
from pathlib import Path


def _try_import_kagglehub():
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            "kagglehub is required for this script. Install it with `pip install kagglehub`."
        ) from exc
    return kagglehub


def _collect_media_files(root_dir):
    media_files = []
    for extension in (
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    ):
        media_files.extend(Path(root_dir).rglob(extension))
    return sorted({path.resolve() for path in media_files})


def _build_manifest(media_paths, output_dir):
    manifest = []
    for index, media_path in enumerate(media_paths):
        suffix = media_path.suffix.lower()
        item_type = "video" if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"} else "image"
        manifest.append(
            {
                "sample_id": f"sample_{index:06d}",
                "item_type": item_type,
                "video_file_path": str(media_path),
                "video_path": str(media_path),
                "video_file": str(media_path),
            }
        )

    manifest_path = Path(output_dir) / "video_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Download a Kaggle video dataset and build a manifest.")
    parser.add_argument(
        "--dataset",
        default="maadaaai/sunny-day-city-road-dash-cam-video-dataset",
        help="Kaggle dataset slug to download.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where the dataset will be copied. Defaults to the KaggleHub download location.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy videos into output-root/videos instead of leaving them in the KaggleHub cache.",
    )
    args = parser.parse_args()

    kagglehub = _try_import_kagglehub()
    download_root = Path(kagglehub.dataset_download(args.dataset)).resolve()

    destination_root = Path(args.output_root).resolve() if args.output_root else download_root
    destination_root.mkdir(parents=True, exist_ok=True)

    source_media_paths = _collect_media_files(download_root)
    if not source_media_paths:
        raise RuntimeError(f"No video or image files were found under {download_root}")

    if args.copy:
        media_dir = destination_root / "videos"
        media_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        for source_path in source_media_paths:
            target_path = media_dir / source_path.name
            shutil.copy2(source_path, target_path)
            copied_paths.append(target_path.resolve())
        source_media_paths = copied_paths

    manifest_path = _build_manifest(source_media_paths, destination_root)

    print(f"Downloaded dataset root: {download_root}")
    print(f"Prepared video root: {destination_root}")
    print(f"Video manifest: {manifest_path}")
    print("Run the model with: --use_video_encoder --use_raw_video --video_raw_root <prepared media root>")


if __name__ == "__main__":
    main()