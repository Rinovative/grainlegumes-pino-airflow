"""
===============================================================================
 split_batch_to_cases.py
===============================================================================
Splits a merged <batch_name>.pt dataset back into individual case_XXXX.pt files.

Input dataset contains:
    - inputs   : (N, C_in, ny, nx)
    - outputs  : (N, C_out, ny, nx)
    - fields   : { "inputs": [...], "outputs": [...] }

This script:
    1. Loads the merged batch dataset.
    2. Iterates over cases (N dimension).
    3. Reconstructs per-case dictionaries:
           {
             "input_fields":  {field_name: (ny, nx)},
             "output_fields": {field_name: (ny, nx)},
           }
    4. Saves each case as case_XXXX.pt into data/raw/<batch_name>/cases
===============================================================================
"""

import torch
from src import common
from tqdm import tqdm


def split_batch_to_cases(
    batch_name: str,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Split a merged PINO/FNO batch dataset into individual case_XXXX.pt files.

    Parameters
    ----------
    batch_name : str
        Name of the batch dataset.
    overwrite : bool
        If True, existing case files will be overwritten.
    verbose : bool
        If True, prints progress and structure info.

    Returns
    -------
    dict
        Summary information.

    """
    log = []
    # Training datasets are in TRAIN_ROOT
    src_dataset_path = common.paths.get_train_root() / batch_name / f"{batch_name}.pt"
    dst_cases_dir = common.paths.get_data_root() / batch_name / "cases"
    dst_cases_dir.mkdir(parents=True, exist_ok=True)

    if not src_dataset_path.exists():
        msg = f"Dataset not found: {src_dataset_path}"
        raise RuntimeError(msg)

    dataset = torch.load(src_dataset_path, map_location="cpu", weights_only=False)

    inputs = dataset["inputs"]  # (N, C_in, ny, nx)
    outputs = dataset["outputs"]  # (N, C_out, ny, nx)
    in_fields = dataset["fields"]["inputs"]
    out_fields = dataset["fields"]["outputs"]

    n_cases = inputs.shape[0]

    log.append(f"Splitting dataset: {src_dataset_path}")
    log.append(f"Number of cases: {n_cases}")
    log.append(f"Destination: {dst_cases_dir}")

    if verbose:
        log.append(f"Input fields:  {in_fields}")
        log.append(f"Output fields: {out_fields}")

    pbar = tqdm(
        range(n_cases),
        desc=f"Splitting {batch_name}",
        unit="case",
        disable=not verbose,
    )

    for i in pbar:
        case_path = dst_cases_dir / f"case_{i:04d}.pt"

        if case_path.exists() and not overwrite:
            continue

        input_fields = {name: inputs[i, j].numpy() for j, name in enumerate(in_fields)}

        output_fields = {name: outputs[i, j].numpy() for j, name in enumerate(out_fields)}

        case_dict = {
            "input_fields": input_fields,
            "output_fields": output_fields,
        }

        torch.save(case_dict, case_path)

    pbar.close()

    log.append(f"Cases written: {n_cases}")

    return {
        "batch_name": batch_name,
        "n_cases": n_cases,
        "cases_dir": dst_cases_dir,
        "log": log,
    }


if __name__ == "__main__":
    result = split_batch_to_cases(
        "lhs_var160_seed5001",
        overwrite=False,
        verbose=True,
    )
    for line in result["log"]:
        print(line)
