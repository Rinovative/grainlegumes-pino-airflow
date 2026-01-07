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

Each case file contains:
    - input_fields:   x, y, kappa* tensors (log10), phi, p_bc
    - output_fields:  p, u, v, U
    - meta:           simulation metadata

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
from tqdm import tqdm

# ============================================================================
# Field definitions (single source of truth)
# ============================================================================

# coordinate fields
COORD_FIELDS = ("x", "y")

# permeability tensor prefix
KAPPA_PREFIX = "br.kappa"

# scalar input fields exported as volume fields
INPUT_SCALAR_FIELDS = {
    "int4(x,y)": "phi",
    "int5(x,y)": "p_bc",
}

# output fields
OUTPUT_FIELDS = {
    "p": "p",
    "u": "u",
    "v": "v",
    "br.U": "U",
}


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
    proj_root = Path(__file__).resolve().parents[2]
    gen_data_dir = proj_root / "data_generation" / "data"

    proc_dir = gen_data_dir / "processed" / batch_name
    raw_dir = gen_data_dir / "raw" / batch_name
    meta_dir = gen_data_dir / "meta"

    out_root = proj_root / "data" / "raw"
    out_batch = out_root / batch_name
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

        return df, meta

    def build_fields(
        df: pd.DataFrame,
        nx: int,
        ny: int,
        dropped_input: list[str],
        dropped_output: list[str],
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
        dropped_input : list[str]
            List of input fields to drop.
        dropped_output : list[str]
            List of output fields to drop.

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
        # Permeability tensor
        #   - diagonal components (xx, yy, zz): log10
        #   - off-diagonal components (xy, yx, ...): relative scaling
        # --------------------------------------------------------------
        eps = 1e-12

        # cache diagonals for normalization
        diag = {}
        for col in (c for c in df.columns if c.startswith(KAPPA_PREFIX)):
            name = col.replace("br.", "")
            ij = name[-2:]
            if ij[0] == ij[1]:
                diag[ij[0]] = df[col].to_numpy().reshape(ny, nx)

        for col in (c for c in df.columns if c.startswith(KAPPA_PREFIX)):
            name = col.replace("br.", "")
            if name in dropped_input:
                continue

            raw = df[col].to_numpy().reshape(ny, nx)
            ij = name[-2:]

            if ij[0] == ij[1]:
                # diagonal
                input_fields[name] = np.log10(raw + eps)
            else:
                # off-diagonal: relative scaling
                kii = diag.get(ij[0])
                kjj = diag.get(ij[1])
                if kii is None or kjj is None:
                    # fallback (should not happen if tensor is complete)
                    input_fields[name] = raw
                else:
                    input_fields[name] = raw / np.sqrt(kii * kjj + eps)

        # --------------------------------------------------------------
        # Scalar input fields (phi, p_bc)
        # --------------------------------------------------------------
        for csv_name, field_name in INPUT_SCALAR_FIELDS.items():
            if csv_name in df.columns:
                input_fields[field_name] = df[csv_name].to_numpy().reshape(ny, nx)

        # --------------------------------------------------------------
        # Outputs
        # --------------------------------------------------------------
        for csv_name, field_name in OUTPUT_FIELDS.items():
            if csv_name in df.columns and field_name not in dropped_output:
                output_fields[field_name] = df[csv_name].to_numpy().reshape(ny, nx)

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

    dropped_input: list[str] = []
    dropped_output: list[str] = []

    dropped_input = [
        col.replace("br.", "") for col in df_first.columns if col.startswith(KAPPA_PREFIX) and not np.any(df_first[col].to_numpy().reshape(ny, nx))
    ]

    for csv_name, field_name in OUTPUT_FIELDS.items():
        if csv_name in df_first.columns and not np.any(df_first[csv_name].to_numpy().reshape(ny, nx)):
            dropped_output.append(field_name)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    preview_in, preview_out = build_fields(
        df_first,
        nx,
        ny,
        dropped_input,
        dropped_output,
    )

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

        input_fields, output_fields = build_fields(
            df,
            nx,
            ny,
            dropped_input,
            dropped_output,
        )

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
    result = build_batch_dataset("lhs_var120_seed4001", verbose=True)
    for line in result["log"]:
        print(line)
