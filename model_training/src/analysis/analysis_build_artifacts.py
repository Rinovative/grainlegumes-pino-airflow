"""Generate analysis artifacts for all models and datasets."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

from src import analysis
from src.analysis import evaluation

if TYPE_CHECKING:
    from collections.abc import Iterable

# ======================================================================
# Global config
# ======================================================================

# GPU aktiv lassen (leer setzen, falls CPU gewuenscht)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "model_training" / "data" / "processed"

ID_DATASET = "lhs_var80_seed3001"
OOD_DATASETS = ["lhs_var120_seed4001"]

# Anzahl Cases begrenzen (None = alle)
MAX_CASES: int | None = None
# ======================================================================
# Utilities
# ======================================================================


def cleanup_gpu() -> None:
    """Aggressive GPU / memory cleanup after each model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def iter_model_dirs(root: Path) -> Iterable[str]:
    """Iterate over all model directories in the given root."""
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue

        if (p / "best_model_state_dict.pt").exists() or (p / "model_state_dict.pt").exists():
            yield p.name


def run_or_load_artifacts(
    *,
    model_name: str,
    dataset_name: str,
    max_cases: int | None,
) -> pd.DataFrame:
    """Load or generate artifacts for one (model, dataset) pair."""
    run_dir = PROCESSED_ROOT / model_name
    best_ckpt = run_dir / "best_model_state_dict.pt"
    last_ckpt = run_dir / "model_state_dict.pt"

    if best_ckpt.exists():
        checkpoint_path = best_ckpt
    elif last_ckpt.exists():
        checkpoint_path = last_ckpt
    else:
        msg = f"No checkpoint found in {run_dir}"
        raise FileNotFoundError(msg)
    dataset_path = RAW_ROOT / dataset_name / "cases"

    save_root = run_dir / "analysis" / "id" if dataset_name == ID_DATASET else run_dir / "analysis" / "ood" / dataset_name
    npz_dir = save_root / "npz"
    parquet_path = save_root / f"{dataset_name}.parquet"

    print(f"[RUN] {model_name} | {dataset_name}")
    print(f"      checkpoint={checkpoint_path}")
    print(f"      save_root={save_root}")

    # --------------------------------------------------
    # HARD SKIP: artifacts already exist
    # --------------------------------------------------
    if parquet_path.exists():
        print(f"[LOAD] {model_name} | {dataset_name} (parquet)")
        return pd.read_parquet(parquet_path)

    if npz_dir.exists() and any(npz_dir.glob("*.npz")):
        print(f"[SKIP] {model_name} | {dataset_name} (npz exists, no parquet)")
        return pd.DataFrame()

    try:
        model, loader, processor, device = analysis.analysis_interference.load_inference_context(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            batch_size=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] {model_name} | {dataset_name}")
        print(f"       Reason: {type(e).__name__}: {e}")
        return pd.DataFrame()

    df, _ = analysis.analysis_artifacts.generate_artifacts(
        model=model,
        loader=loader,
        processor=processor,
        device=device,
        save_root=save_root,
        dataset_name=dataset_name,
        max_cases=max_cases,
    )

    # explizit loeschen
    del model, loader, processor
    cleanup_gpu()

    return df


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Generate all analysis artifacts for all models and datasets."""
    model_names = list(iter_model_dirs(PROCESSED_ROOT))
    print(f"[INFO] Found {len(model_names)} models")

    for model_name in model_names:
        print(f"\n=== {model_name} ===")

        # -----------------
        # ID
        # -----------------
        df_raw_id = run_or_load_artifacts(
            model_name=model_name,
            dataset_name=ID_DATASET,
            max_cases=MAX_CASES,
        )
        if df_raw_id.empty:
            print(f"[SKIP] {model_name} | ID evaluation skipped")
            continue
        _ = evaluation.evaluation_dataframe.build_eval_df(df_raw_id)

        # -----------------
        # OOD
        # -----------------
        for ood in OOD_DATASETS:
            df_raw_ood = run_or_load_artifacts(
                model_name=model_name,
                dataset_name=ood,
                max_cases=MAX_CASES,
            )

            if df_raw_ood.empty:
                print(f"[SKIP] {model_name} | OOD {ood} skipped")
                continue

            _ = evaluation.evaluation_dataframe.build_eval_df(df_raw_ood)

        cleanup_gpu()

    print("\n[DONE] All artifacts generated.")


if __name__ == "__main__":
    main()
