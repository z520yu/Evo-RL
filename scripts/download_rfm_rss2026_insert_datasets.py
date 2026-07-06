#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


OUT_ROOT = Path("/home/lenovo/桌面/Posttraining-RFM-RSS2026_insert_datasets")

PHASE2_REPO = "Posttraining-RFM-RSS2026/Challenge-phase2-rollouts-dataset"
PHASE2_ALLOW = [
    "HIL_generalist_final_insert-mouse-battery_*/*",
    "HIL_specialist_final_insert-mouse-battery_*/*",
]

PHASE1_REPO = "Posttraining-RFM-RSS2026/Challenge-phase1-dataset"
PHASE1_ALLOW = [
    "insert-mouse-battery/*",
]


def download_one(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== downloading {repo_id} ===", flush=True)
    print(f"local_dir={local_dir}", flush=True)
    print("allow_patterns:", *allow_patterns, sep="\n  ", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        resume_download=True,
        max_workers=8,
    )


def main() -> None:
    download_one(PHASE2_REPO, OUT_ROOT / "phase2_rollouts_insert", PHASE2_ALLOW)
    download_one(PHASE1_REPO, OUT_ROOT / "phase1_insert_mouse_battery", PHASE1_ALLOW)
    print("\nDone.", flush=True)
    print(f"Output root: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
