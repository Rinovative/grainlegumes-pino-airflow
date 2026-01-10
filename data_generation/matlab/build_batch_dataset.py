"""
===============================================================================
 build_batch_dataset.
===============================================================================
Author:  Rino M. Albertin
Date:    2025-10-28
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Reads raw COMSOL simulation outputs and converts them into structured PyTorch
`.pt` case files for PINO/FNO training and evaluation.

Permeability tensors are handled via a central schema:
  - Automatic detection of 2D vs 3D problems
  - Canonical internal tensor representation
  - Symmetric COMSOL components (e.g. kappaxy, kappayx) are averaged
  - Diagonal components are stored in log10-space
  - Off-diagonal components are stored in normalized, dimensionless form

Each case file contains:
  - input_fields:
      x, y,
      kxx, kyy (, kzz),
      kxy (, kxz, kyz),
      phi, p_bc
  - output_fields:
      p, u, v, U
  - meta:

Note:
----
p_bc is a boundary condition exported by COMSOL as a volume field.
Values are zero everywhere except on the prescribed boundary.
This encoding is intentional to keep a purely field-based PINO input.
===============================================================================

"""  # noqa: D205

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.schema.schema_fields import COORD_FIELDS, OUTPUT_FIELDS, SCALAR_INPUT_FIELDS
from src.schema.schema_kappa import resolve_internal_to_present_sources
from tqdm import tqdm

# =============================================================================
# COMSOL name prefix
# =============================================================================
COMSOL_PREFIX = "br."


def _prune_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Prune unnecessary information from the meta dictionary.

    Parameters
    ----------
    meta : dict[str, Any]
        Original metadata dictionary.

    Returns
    -------
    dict[str, Any]
        Pruned metadata dictionary.

    """
    gen = meta.get("generator")
    if not isinstance(gen, dict):
        return meta

    def _clean_block(block: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively clean a block of the meta dictionary.

        Parameters
        ----------
        block : dict[str, Any]
            Block to clean.

        Returns
        -------
            dict[str, Any]
                Cleaned block.

        """
        out: dict[str, Any] = {}
        for k, v in block.items():
            # drop RNG / state / reproducibility info
            if "state" in k.lower():
                continue

            if isinstance(v, dict):
                cleaned = _clean_block(v)
                if cleaned:
                    out[k] = cleaned
            # keep ONLY numeric values
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = v
        return out

    meta["generator"] = _clean_block(gen)
    return meta


def build_batch_dataset(batch_name: str, verbose: bool = False) -> dict:  # noqa: C901, PLR0915
    """
    Build a batch dataset from raw COMSOL outputs.

    Parameters
    ----------
    batch_name : str
        Name of the batch to process (corresponds to folder names).
    verbose : bool, optional
        If True, prints additional information, by default False.

    Returns
    -------
    dict
        Summary of the dataset building process.

    """
    log: list[str] = []

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_GENERATION = PROJECT_ROOT / "data_generation" / "data"
    DATA_RAW = PROJECT_ROOT / "data" / "raw"

    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    import os

    DATA_ROOT = Path(os.environ.get("TMP_DATA_ROOT", "/home/rino.albertin/workspace/tmp_data"))

    DATA_GENERATION = DATA_ROOT / "temp_data_generation"
    DATA_RAW = DATA_ROOT / "temp_data" / "temp_raw"
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================
    # ==== TEMPORARY: DOCKER/GPU NOT AVAILABLE ========================

    proc_dir = DATA_GENERATION / "processed" / batch_name
    raw_dir = DATA_GENERATION / "raw" / batch_name
    meta_dir = DATA_GENERATION / "meta"
    out_batch = DATA_RAW / batch_name

    cases_dir = out_batch / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    log.append(f"Processing batch: {batch_name}")

    # ------------------------------------------------------------------
    # Case discovery
    # ------------------------------------------------------------------
    json_files = sorted(raw_dir.glob("case_*.json"))
    csv_files = sorted(proc_dir.glob("case_*_sol.csv"))

    json_names = {f.stem for f in json_files}
    csv_names = {f.stem.replace("_sol", "") for f in csv_files}
    common = sorted(json_names.intersection(csv_names))

    if not common:
        msg = f"No valid matching cases found for {batch_name}"
        raise RuntimeError(msg)

    log.append(f"Found {len(common)} matching cases.")

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def load_case(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load a single case from CSV and JSON files.

        Parameters
        ----------
        csv_path : Path
            Path to the CSV file.
        meta_path : Path
            Path to the JSON metadata file.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            DataFrame with case data and metadata dictionary.

        """
        with meta_path.open() as f:
            meta = json.load(f)

        meta = _prune_meta(meta)

        with csv_path.open() as f:
            lines = f.readlines()

        header_line = [line for line in lines if line.strip().startswith("%")][-1]
        sep = ";"
        header = header_line[1:].split(";")

        header = [h.strip() for h in header]

        if header.count("y") > 1:
            idx = [i for i, h in enumerate(header) if h == "y"][1]
            header[idx] = "y (on)"

        df = pd.read_csv(
            csv_path,
            comment="%",
            sep=sep,
            names=header,
            index_col=False,
            skip_blank_lines=True,
        )
        df = df.copy()
        df.columns = [c.removeprefix(COMSOL_PREFIX) for c in df.columns]

        return df, meta

    def build_fields(
        df: pd.DataFrame,
        nx: int,
        ny: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Build input and output fields from a case DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing case data.
        nx : int
            Number of grid points in x direction.
        ny : int
            Number of grid points in y direction.

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
            Dictionaries of input and output fields.

        """
        input_fields: dict[str, np.ndarray] = {}
        output_fields: dict[str, np.ndarray] = {}

        # --------------------------------------------------------------
        # Coordinates
        # --------------------------------------------------------------
        for c in COORD_FIELDS:
            if c in df.columns:
                input_fields[c] = df[c].to_numpy().reshape(ny, nx)

        # --------------------------------------------------------------
        # Permeability tensor (schema-driven, canonical representation)
        #
        # The permeability tensor is constructed using the central
        # kappa schema, which defines:
        #   - the problem dimensionality (2D vs 3D),
        #   - the canonical internal component order,
        #   - and the mapping from COMSOL-exported components to
        #     internal symmetric tensor components.
        #
        # Symmetric COMSOL components (e.g. kappaxy, kappayx) are
        # averaged to enforce numerical symmetry and robustness.
        #
        # Diagonal components are stored in log10-space.
        # Off-diagonal components are stored in dimensionless,
        # normalized form using sqrt(k_ii * k_jj).
        # --------------------------------------------------------------
        eps = 1e-12

        # Collect all available COMSOL permeability components
        # (strip the COMSOL_PREFIX 'br.' to match schema naming)
        available_kappa = [c for c in df.columns if c.startswith("kappa")]

        # Determine which kappa components are physically non-zero
        KAPPA_ZERO_TOL = 1e-14

        nonzero_kappa = []
        for name in available_kappa:
            field = df[name].to_numpy()
            if np.any(np.abs(field) > KAPPA_ZERO_TOL):
                nonzero_kappa.append(name)

        # Let the schema decide:
        # - dimensionality (2D vs 3D)
        # - canonical internal component order
        # - mapping from internal components to COMSOL sources
        kappa_mapping = resolve_internal_to_present_sources(available_kappa, nonzero_kappa)

        if not kappa_mapping:
            msg = "No permeability components found in dataset"
            raise RuntimeError(msg)

        for internal_name, source_fields in kappa_mapping.items():
            # Load all COMSOL source fields contributing to this internal component
            tensors = [df[src].to_numpy().reshape(ny, nx) for src in source_fields]

            # Average symmetric components if multiple sources are present
            raw = sum(tensors) / len(tensors)

            if internal_name in ("kxx", "kyy", "kzz"):
                # Diagonal components: stored in log10-space
                input_fields[internal_name] = np.log10(raw + eps)
            else:
                # Off-diagonal components: dimensionless, normalized by
                # sqrt(k_ii * k_jj) to ensure scale consistency
                i, j = internal_name[1], internal_name[2]
                kii = input_fields[f"k{i}{i}"]
                kjj = input_fields[f"k{j}{j}"]
                input_fields[internal_name] = raw / np.sqrt(10**kii * 10**kjj + eps)

        # --------------------------------------------------------------
        # Scalar input fields (phi, p_bc)
        # --------------------------------------------------------------
        for internal, col in SCALAR_INPUT_FIELDS.items():
            if col not in df.columns:
                msg = f"Missing required input field '{col}'"
                raise KeyError(msg)
            input_fields[internal] = df[col].to_numpy().reshape(ny, nx)

        # --------------------------------------------------------------
        # Outputs
        # --------------------------------------------------------------
        for name in OUTPUT_FIELDS:
            if name not in df.columns:
                msg = f"Missing required output field '{name}'"
                raise KeyError(msg)
            output_fields[name] = df[name].to_numpy().reshape(ny, nx)

        return input_fields, output_fields

    # ------------------------------------------------------------------
    # Drop detection (reference case)
    # ------------------------------------------------------------------
    df_first, meta_first = load_case(
        proc_dir / f"{common[0]}_sol.csv",
        raw_dir / f"{common[0]}.json",
    )

    nx = meta_first["geometry"]["nx"]
    ny = meta_first["geometry"]["ny"]

    if verbose:
        print("\n[DEBUG] df_first.columns:")
        for col in df_first.columns:
            print(f"  - {col}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    preview_in, preview_out = build_fields(df_first, nx, ny)

    if verbose:
        print("\nExample structure for first case:")
        print("--------------------------------------------------")
        print("input_fields:")
        for k, v in preview_in.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("output_fields:")
        for k, v in preview_out.items():
            print(f"  {k:10s}  shape={v.shape}, dtype={v.dtype}")
        print("meta keys:", list(meta_first.keys()))
        print("--------------------------------------------------\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    pbar = tqdm(
        total=len(common),
        desc=f"Building {batch_name}",
        unit="file",
        disable=not verbose,
    )

    for name in common:
        df, meta = load_case(
            proc_dir / f"{name}_sol.csv",
            raw_dir / f"{name}.json",
        )

        nx = meta["geometry"]["nx"]
        ny = meta["geometry"]["ny"]

        input_fields, output_fields = build_fields(df, nx, ny)

        torch.save(
            {
                "input_fields": input_fields,
                "output_fields": output_fields,
                "meta": meta,
            },
            cases_dir / f"{name}.pt",
        )

        pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------------
    # Batch meta
    # ------------------------------------------------------------------
    meta_saved = False
    meta_json = meta_dir / f"{batch_name}.json"
    meta_csv = meta_dir / f"{batch_name}.csv"

    if meta_json.exists() and meta_csv.exists():
        with meta_json.open() as f:
            meta_struct = {
                "json": json.load(f),
                "csv": pd.read_csv(meta_csv).to_dict(orient="list"),
            }
        torch.save(meta_struct, out_batch / "meta.pt")
        meta_saved = True
        log.append("Saved meta.pt")
    else:
        log.append("No meta files found.")

    return {
        "batch_name": batch_name,
        "n_cases": len(common),
        "cases_dir": cases_dir,
        "out_batch": out_batch,
        "meta_saved": meta_saved,
        "log": log,
    }


if __name__ == "__main__":
    result = build_batch_dataset("lhs_var160_seed5001", verbose=True)
    for line in result["log"]:
        print(line)
