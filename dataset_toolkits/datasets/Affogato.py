import os
import glob
import hashlib
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--affogato_base_dir",
        type=str,
        default="/path/to/affogato_raw",
        help="Base directory containing AffoGato dataset",
    )
    parser.add_argument(
        "--trellis_data_dir",
        type=str,
        default=os.path.expanduser("~/datasets/trellis"),
        help="TRELLIS dataset directory containing ObjaverseXL metadata",
    )


def get_metadata(affogato_base_dir, trellis_data_dir, **kwargs):
    """
    Scan AffoGato dataset and build metadata following ObjaverseXL convention.

    Uses ObjaverseXL sha256 when available (for data used in TRELLIS training),
    otherwise computes SHA256(UID) for new data.

    Splits are determined by removing TRELLIS training data from test/val sets.

    Returns:
        pd.DataFrame with columns: sha256, file_identifier, aesthetic_score, captions, split, tar_path, local_path
    """
    affogato_base = Path(affogato_base_dir)
    trellis_base = Path(trellis_data_dir)

    tar_base = affogato_base / "gobjaverse_reduced"
    glb_base = affogato_base / "hf-objaverse-v1" / "glbs"

    if not tar_base.exists():
        raise FileNotFoundError(f"AffoGato directory not found: {tar_base}")

    print("Scanning AffoGato dataset...")

    tar_pattern = str(tar_base / "*" / "*.tar.gz")
    tar_files = glob.glob(tar_pattern)

    print(f"Found {len(tar_files)} tar.gz files")

    print("Loading ObjaverseXL metadata...")

    objaverse_xl_sketchfab = pd.read_csv(trellis_base / "objaversexl_sketchfab" / "metadata.csv")
    objaverse_xl_github = pd.read_csv(trellis_base / "objaversexl_github" / "metadata.csv")

    objaverse_xl_sketchfab["uid"] = objaverse_xl_sketchfab["file_identifier"].str.extract(
        r"/([a-f0-9]{32})$"
    )
    objaverse_xl_github["uid_candidates"] = objaverse_xl_github["file_identifier"].str.findall(
        r"[a-f0-9]{32}"
    )
    objaverse_xl_github["uid"] = objaverse_xl_github["uid_candidates"].apply(
        lambda x: x[0] if len(x) > 0 else None
    )

    uid_to_sha256 = {}
    uid_to_aesthetic = {}

    for _, row in objaverse_xl_github.iterrows():
        if pd.notna(row["uid"]):
            uid_to_sha256[row["uid"]] = row["sha256"]
            uid_to_aesthetic[row["uid"]] = row["aesthetic_score"]

    for _, row in objaverse_xl_sketchfab.iterrows():
        if pd.notna(row["uid"]):
            uid_to_sha256[row["uid"]] = row["sha256"]
            uid_to_aesthetic[row["uid"]] = row["aesthetic_score"]

    print(f"ObjaverseXL UID mappings: {len(uid_to_sha256)} UIDs")

    trellis_trained_uids = {uid for uid, aes in uid_to_aesthetic.items() if aes >= 5.5}
    print(f"TRELLIS training data: {len(trellis_trained_uids)} UIDs (aesthetic >= 5.5)")

    split_dir = Path(__file__).parent.parent / "metadata"

    train_uids = set()
    val_uids = set()
    test_uids = set()

    with open(split_dir / "affogato_train.txt") as f:
        train_uids = {line.strip().split("@")[0] for line in f if line.strip()}

    with open(split_dir / "affogato_val.txt") as f:
        val_uids = {line.strip().split("@")[0] for line in f if line.strip()}

    with open(split_dir / "affogato_test.txt") as f:
        test_uids = {line.strip().split("@")[0] for line in f if line.strip()}

    print(
        f"Original splits - Train: {len(train_uids)}, Val: {len(val_uids)}, Test: {len(test_uids)}"
    )

    test_in_trellis = test_uids & trellis_trained_uids
    val_in_trellis = val_uids & trellis_trained_uids

    new_train = train_uids | test_in_trellis | val_in_trellis
    new_val = val_uids - trellis_trained_uids
    new_test = test_uids - trellis_trained_uids

    print(
        f"Adjusted splits - Train: {len(new_train)} (+{len(test_in_trellis) + len(val_in_trellis)}), "
        f"Val: {len(new_val)} (-{len(val_in_trellis)}), Test: {len(new_test)} (-{len(test_in_trellis)})"
    )

    uid_to_split = {}
    for uid in new_train:
        uid_to_split[uid] = "train"
    for uid in new_val:
        uid_to_split[uid] = "val"
    for uid in new_test:
        uid_to_split[uid] = "test"

    records = []

    for tar_path in tqdm(tar_files, desc="Building metadata"):
        uid = os.path.basename(tar_path).replace(".tar.gz", "")

        if uid in uid_to_sha256:
            sha256 = uid_to_sha256[uid]
        else:
            sha256 = hashlib.sha256(uid.encode()).hexdigest()

        aesthetic_score = uid_to_aesthetic.get(uid, None)

        split = uid_to_split.get(uid, "train")

        subdir = uid[:3] + "-" + uid[3:6]
        glb_path = glb_base / subdir / f"{uid}.glb"

        if not glb_path.exists():
            for potential_subdir in glb_base.iterdir():
                if potential_subdir.is_dir():
                    potential_glb = potential_subdir / f"{uid}.glb"
                    if potential_glb.exists():
                        glb_path = potential_glb
                        break

        tar_rel_path = os.path.relpath(tar_path, affogato_base)
        glb_rel_path = os.path.relpath(glb_path, affogato_base) if glb_path.exists() else None

        records.append(
            {
                "sha256": sha256,
                "file_identifier": uid,
                "aesthetic_score": aesthetic_score,
                "captions": None,
                "split": split,
                "tar_path": tar_rel_path,
                "local_path": glb_rel_path,
            }
        )

    metadata = pd.DataFrame.from_records(records)

    print(f"\nGenerated metadata for {len(metadata)} objects")
    print(f"  With ObjaverseXL sha256: {metadata['aesthetic_score'].notna().sum()}")
    print(f"  With computed sha256: {metadata['aesthetic_score'].isna().sum()}")
    print(
        f"  Split distribution - Train: {(metadata['split'] == 'train').sum()}, "
        f"Val: {(metadata['split'] == 'val').sum()}, Test: {(metadata['split'] == 'test').sum()}"
    )

    return metadata


def _process_instance_wrapper(args):
    """
    Top-level wrapper function for ProcessPoolExecutor (must be pickle-able).

    Args:
        args: Tuple of (metadatum_dict, affogato_base_dir, output_dir, func)

    Returns:
        Processing result record or None
    """
    metadatum, affogato_base_dir, output_dir, func = args

    try:
        file_identifier = metadatum["file_identifier"]
        sha256 = metadatum["sha256"]
        tar_rel_path = metadatum["tar_path"]
        glb_rel_path = metadatum.get("local_path")

        tar_path = os.path.join(affogato_base_dir, tar_rel_path)
        glb_path = os.path.join(affogato_base_dir, glb_rel_path) if glb_rel_path else None

        record = func(file_identifier, sha256, tar_path, glb_path, output_dir)
        return record

    except Exception as e:
        print(f"Error processing object {metadatum.get('file_identifier', 'unknown')}: {e}")
        return None


def foreach_instance(
    metadata, output_dir, func, max_workers=None, desc="Processing objects", affogato_base_dir=None
) -> pd.DataFrame:
    """
    Process each instance in parallel using the provided function.

    Args:
        metadata: DataFrame with columns including 'file_identifier', 'sha256', 'tar_path', 'local_path'
        output_dir: Base output directory
        func: Function to process each instance. Should accept (file_identifier, sha256, tar_path, glb_path, output_dir)
              and return a dict record or None
        max_workers: Number of parallel workers (defaults to CPU count)
        desc: Description for progress bar

    Returns:
        DataFrame of processing records
    """
    metadata_records = metadata.to_dict("records")

    if affogato_base_dir is None:
        raise ValueError("foreach_instance requires affogato_base_dir (raw Affogato data root)")

    worker_args = [(metadatum, affogato_base_dir, output_dir, func) for metadatum in metadata_records]

    records = []
    max_workers = max_workers or os.cpu_count()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance_wrapper, args): args[0]["file_identifier"]
                for args in worker_args
            }

            with tqdm(total=len(futures), desc=desc) as pbar:
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        if record is not None:
                            records.append(record)
                    except Exception as e:
                        uid = futures[future]
                        print(f"Error processing {uid}: {e}")
                    finally:
                        pbar.update(1)

    except Exception as e:
        print(f"Error happened during processing: {e}")
        import traceback

        traceback.print_exc()

    return pd.DataFrame.from_records(records)
