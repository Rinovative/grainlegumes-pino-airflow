# compute_artifacts_all_models.py
import os
from pathlib import Path

from src import analysis

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU

# ======================================================================
# Config
# ======================================================================
ID_DATASET = "lhs_var80_seed3001"
OOD_DATASET = "lhs_var120_seed4001"

THIS_DIR = Path(__file__).resolve().parent  # .../model_training/notebooks
RAW_ROOT = (THIS_DIR / "../../data/raw").resolve()
PROCESSED_ROOT = (THIS_DIR / "../data/processed").resolve()

CHECKPOINT_NAME = "best_model_state_dict.pt"
BATCH_SIZE = 1
FORCE = False  # True = alles neu rechnen, False = resumable (skip wenn vorhanden)


# ======================================================================
# Helpers
# ======================================================================
def discover_models(processed_root: Path) -> list[str]:
    models: list[str] = []
    for p in sorted(processed_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / CHECKPOINT_NAME).exists():
            models.append(p.name)
    if not models:
        msg = f"Keine Modelle gefunden in {processed_root} (mit {CHECKPOINT_NAME})"
        raise RuntimeError(msg)
    return models


def ensure_artifacts(*, model_name: str, dataset_name: str) -> None:
    run_dir = PROCESSED_ROOT / model_name
    checkpoint_path = run_dir / CHECKPOINT_NAME

    dataset_path = RAW_ROOT / dataset_name / "cases"
    if not dataset_path.exists():
        msg = f"Dataset nicht gefunden: {dataset_path}"
        raise FileNotFoundError(msg)

    # Zielordner bestimmen (OHNE ihn direkt zu erstellen)
    save_root = run_dir / "analysis" / "id" if dataset_name == ID_DATASET else run_dir / "analysis" / "ood" / dataset_name

    # WICHTIG: zuerst Ordner-Existenz pruefen
    if save_root.exists() and not FORCE:
        print(f"[SKIP] {model_name} | {dataset_name} (folder exists: {save_root})")
        return

    # erst jetzt erstellen
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"[RUN ] {model_name} | {dataset_name} -> creating artifacts")

    model, loader, processor, device = analysis.analysis_interference.load_inference_context(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        batch_size=BATCH_SIZE,
    )

    analysis.analysis_artifacts.generate_artifacts(
        model=model,
        loader=loader,
        processor=processor,
        device=device,
        save_root=save_root,
        dataset_name=dataset_name,
    )


# ======================================================================
# Main
# ======================================================================
def main():
    models = discover_models(PROCESSED_ROOT)
    print(f"[INFO] Gefundene Modelle: {len(models)}")

    ok = 0
    failed = 0

    for model_name in models:
        print(f"\n=== {model_name} ===")
        try:
            ensure_artifacts(model_name=model_name, dataset_name=ID_DATASET)
            ensure_artifacts(model_name=model_name, dataset_name=OOD_DATASET)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed: {model_name} | {type(e).__name__}: {e}")

    print(f"\n[INFO] Done. OK: {ok} | Failed: {failed}")


if __name__ == "__main__":
    main()
