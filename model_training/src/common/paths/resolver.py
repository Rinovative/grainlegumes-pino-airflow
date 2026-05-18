"""
Central path resolver for the project.

Handles:
- Reading PROJECT_ROOT, STORAGE_ROOT, DATA_ROOT, GEN_ROOT and TRAIN_ROOT
- Providing stable repo-relative and storage-relative paths
- Resolving dataset names to dataset directories
- Resolving run output directories
- Ensuring Docker, notebook and cluster execution consistency

Environment variables (set by Docker or host):
- PROJECT_ROOT: repository root inside Docker
- STORAGE_ROOT: mounted external storage root
- DATA_ROOT: shared storage data root
- GEN_ROOT: data-generation storage root
- TRAIN_ROOT: model-training storage root

TODO: Full implementation in Phase 1 continuation
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory from environment or default."""
    root = os.environ.get("PROJECT_ROOT")
    if root:
        return Path(root)
    # Fallback: navigate up from this file
    return Path(__file__).parent.parent.parent.parent.parent


def get_storage_root() -> Path:
    """Get the storage root directory from environment or default."""
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root)
    # Fallback: sibling directory to repository
    return get_project_root().parent / "storage"


def get_data_root() -> Path:
    """Get the data root directory from environment or default."""
    root = os.environ.get("DATA_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data"


def get_train_root() -> Path:
    """Get the training data root directory from environment or default."""
    root = os.environ.get("TRAIN_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data_training"


def get_gen_root() -> Path:
    """Get the data generation root directory from environment or default."""
    root = os.environ.get("GEN_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data_generation"
