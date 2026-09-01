"""
Evaluation for the three benchmark protocols.

    python eval.py --task recon    --pred_dir results/benchmark/toys4k/Affostruction/1 \
                                   --data_dir trellis_data/toys4k
    python eval.py --task complete --ckpt chrockey/Affostruction \
                                   --data_dir trellis_data/affogato \
                                   --annotations_dir <affogato_raw>/annotations
    python eval.py --task partial  --ckpt chrockey/Affostruction \
                                   --pred_dir results/benchmark/affogato/Affostruction/1 \
                                   --data_dir trellis_data/affogato

`recon` and `partial` consume the per-object predictions written by
`predict.py`; `complete` runs the affordance flow on ground-truth geometry.
Objects come from `manifests/` — evaluation never depends on the local
`metadata.csv`.

Protocols
---------
recon     Volumetric IoU at 64^3 between the predicted sparse structure and the
          ground-truth voxels. With --mesh_metrics, also Chamfer distance and
          F-score@0.05 over 100k-point clouds sampled from the meshes.
complete  Affordance flow on ground-truth voxels; predictions are transferred to
          the raw 16,384-point annotation cloud and scored with aIoU (mean IoU
          over 20 thresholds, GT binarised at 0.5), AUC, SIM and MAE. One query
          per object (the first) by default.
partial   Affordance flow on the *reconstructed* voxels; predicted and GT voxel
          sets are thresholded at 0.1-0.5 and compared as coordinate sets on the
          shared 64^3 grid, giving aIoU and aCD.

Affordance sampling defaults: 50 steps, cfg_strength 1.0, noise_scale 0.1,
zero negative condition. --seed (default 0) is recorded in the result JSON.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
VOXEL_RESOLUTION = 64

ANNOTATION_SCALE = 0.5 / 0.45
ANNOTATION_ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def load_predicted_voxels(pred_dir, sha256):
    """(N, 3) int voxel coords written by predict.py, or None."""
    path = Path(pred_dir) / sha256 / "outputs.npz"
    if not path.exists():
        return None
    with np.load(path) as npz:
        return npz["sparse_structure"] if "sparse_structure" in npz.files else None


def load_annotation(annotations_dir, uid):
    """Raw 16,384-point Affogato annotation in world coordinates."""
    directory = Path(annotations_dir) / uid
    xyzc_path, queries_path = directory / "xyzc.npy", directory / "queries.json"
    if not xyzc_path.exists() or not queries_path.exists():
        return None
    xyzc = np.load(xyzc_path)
    xyz = (ANNOTATION_ROTATION @ (xyzc[:, :3] * ANNOTATION_SCALE).T).T
    with open(queries_path) as f:
        queries = json.load(f)[0]["queries"]
    return {
        "xyz": xyz.astype(np.float32),
        "heatmaps": xyzc[:, 3:].astype(np.float32),
        "queries": queries,
    }


def voxel_logits_to_points(voxel_coords, logits, points):
    """Nearest-voxel lookup: each point takes the logit of the voxel it falls in.

    Voxels the sparse structure never predicted stay at -inf, so points outside
    the object read as probability 0 rather than 0.5.
    """
    grid = np.full((VOXEL_RESOLUTION,) * 3, -np.inf, dtype=np.float32)
    grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = logits
    clamped = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    index = np.floor((clamped + 0.5) * VOXEL_RESOLUTION).astype(np.int64)
    return grid[index[:, 0], index[:, 1], index[:, 2]]


def read_manifest(path):
    """Object list: lines are ``sha256`` or ``sha256,source_uid``.

    Returns a list of (sha256, uid) with uid None when the manifest has no
    second column.
    """
    entries = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            entries.append((parts[0], parts[1] if len(parts) > 1 else None))
    return entries


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "count": int(array.size),
    }


def save_results(path, entry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(path.read_text()) if path.exists() else []
    if isinstance(history, dict):
        history = [history]
    history.append(entry)
    path.write_text(json.dumps(history, indent=2))
    print(f"\nSaved: {path}")


def load_affordance_pipeline(ckpt, device="cuda"):
    from affostruction import AffordancePipeline

    pipeline = AffordancePipeline.from_pretrained(ckpt)
    return pipeline.cuda() if device == "cuda" else pipeline


def sampler_params(args):
    return {
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "noise_scale": args.noise_scale,
        "neg_cond_mode": args.neg_cond,
    }


def eval_recon(args):
    from affostruction.metrics import chamfer_distance_and_fscore, psnr, volumetric_iou

    pred_dir = Path(args.pred_dir)
    voxel_dir = Path(args.data_dir) / "voxels"
    shas = [sha for sha, _ in read_manifest(args.manifest)]

    scores = {name: [] for name in ("iou", "chamfer_distance", "f_score", "psnr", "lpips", "psnr_normal", "lpips_normal")}
    per_sample, missing = {}, 0
    lpips_metric = None
    if args.appearance_metrics:
        from affostruction.metrics import LPIPS

        lpips_metric = LPIPS()
    for sha256 in tqdm(shas, desc="recon"):
        pred_coords = load_predicted_voxels(pred_dir, sha256)
        gt_path = voxel_dir / f"{sha256}.ply"
        if pred_coords is None or not gt_path.exists():
            missing += 1
            continue
        gt_world = trimesh.load(str(gt_path)).vertices
        gt_coords = ((gt_world + 0.5) * VOXEL_RESOLUTION).astype(np.int32)
        iou = volumetric_iou(pred_coords, gt_coords, VOXEL_RESOLUTION)
        scores["iou"].append(iou)
        per_sample[sha256] = {"iou": iou}

        if args.mesh_metrics:
            pred_points_path = pred_dir / sha256 / "points.ply"
            gt_points_path = Path(args.gt_points_dir) / f"{sha256}.ply"
            if pred_points_path.exists() and gt_points_path.exists():
                yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                pred_points = np.asarray(trimesh.load(str(pred_points_path)).vertices) @ yup_to_zup
                gt_points = np.asarray(trimesh.load(str(gt_points_path)).vertices)
                chamfer, fscore = chamfer_distance_and_fscore(
                    pred_points, gt_points, args.fscore_threshold
                )
                scores["chamfer_distance"].append(chamfer)
                scores["f_score"].append(fscore)
                per_sample[sha256].update({"chamfer_distance": chamfer, "f_score": fscore})

        if args.appearance_metrics:
            for kind, suffix in (("", "rgb.png"), ("_normal", "normal.png")):
                pred_image = pred_dir / sha256 / suffix
                gt_image = Path(args.gt_renders_dir) / sha256 / suffix
                if not pred_image.exists() or not gt_image.exists():
                    continue
                pred_array = np.array(Image.open(pred_image).convert("RGB"))
                gt_array = np.array(Image.open(gt_image).convert("RGB"))
                values = {
                    f"psnr{kind}": psnr(pred_array, gt_array),
                    f"lpips{kind}": lpips_metric(pred_array, gt_array),
                }
                for name, value in values.items():
                    scores[name].append(value)
                per_sample[sha256].update(values)

    results = {name: summarize(values) if values else None for name, values in scores.items()}
    results["num_missing"] = missing
    for name, stats in results.items():
        if isinstance(stats, dict):
            print(f"{name:>16}: {stats['mean']:.4f}  (n={stats['count']})")
    save_results(
        args.output or pred_dir / "recon_metrics.json",
        {"pred_dir": str(pred_dir), "data_dir": args.data_dir, "results": results},
    )
    if args.per_sample:
        (pred_dir / "recon_metrics_per_sample.json").write_text(json.dumps(per_sample))
    return results


def eval_complete(args):
    from affostruction.metrics import AUC, AverageIoU, Similarity, TotalMAE

    pipeline = load_affordance_pipeline(args.ckpt)
    objects = read_manifest(args.manifest)
    metrics = {
        "aiou": AverageIoU(num_thresholds=20, target_threshold=0.5),
        "auc": AUC(target_threshold=0.5),
        "sim": Similarity(eps=1e-12),
        "mae": TotalMAE(),
    }

    samples = []
    for sha256, uid in objects:
        if uid is None:
            raise ValueError(
                "the complete-input protocol needs a manifest with 'sha256,uid' lines "
                "so annotations can be located"
            )
        annotation = load_annotation(args.annotations_dir, uid)
        affordance_path = Path(args.data_dir) / "affordances" / sha256 / "affordance.npz"
        if annotation is None or not affordance_path.exists():
            continue
        samples.append((sha256, annotation, affordance_path))
    print(f"{len(samples)} / {len(objects)} objects have annotations")

    scores = {name: [] for name in metrics}
    per_sample = {}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for start in tqdm(range(0, len(samples), args.batch_size), desc="complete"):
        batch = samples[start : start + args.batch_size]
        coords_list, queries = [], []
        for _, annotation, affordance_path in batch:
            with np.load(affordance_path, allow_pickle=True) as npz:
                coords_list.append(torch.from_numpy(npz["coords"].astype(np.int32)))
            queries.append(annotation["queries"][args.query_idx])

        outputs = pipeline.run_batch(coords_list, queries, **sampler_params(args))

        for (sha256, annotation, _), coords, output in zip(batch, coords_list, outputs):
            point_logits = voxel_logits_to_points(
                coords.numpy().astype(np.int64),
                output["logits"].cpu().numpy(),
                annotation["xyz"],
            )
            pred = [torch.from_numpy(point_logits)]
            target = [torch.from_numpy(annotation["heatmaps"][:, args.query_idx])]
            per_sample[sha256] = {}
            for name, metric in metrics.items():
                value = float(metric(pred, target))
                scores[name].append(value)
                per_sample[sha256][name] = value

    results = {name: summarize(values) for name, values in scores.items()}
    for name, stats in results.items():
        print(f"{name.upper():>5}: {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['count']})")
    output_path = args.output or REPO_ROOT / "results" / "affordance_complete.json"
    save_results(
        output_path,
        {
            "checkpoint": args.ckpt,
            "data_dir": args.data_dir,
            "query_idx": args.query_idx,
            "sampler": sampler_params(args),
            "seed": args.seed,
            "results": results,
        },
    )
    if args.per_sample:
        Path(str(output_path).replace(".json", "_per_sample.json")).write_text(
            json.dumps(per_sample)
        )
    return results


def eval_partial(args):
    from affostruction.metrics import VoxelAffordanceMetrics

    pipeline = load_affordance_pipeline(args.ckpt)
    metric = VoxelAffordanceMetrics(
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], voxel_resolution=VOXEL_RESOLUTION
    )
    pred_dir = Path(args.pred_dir)

    jobs = []
    for sha256, _ in read_manifest(args.manifest):
        pred_coords = load_predicted_voxels(pred_dir, sha256)
        affordance_path = Path(args.data_dir) / "affordances" / sha256 / "affordance.npz"
        if pred_coords is None or not affordance_path.exists():
            continue
        with np.load(affordance_path, allow_pickle=True) as npz:
            queries = [str(q) for q in npz["query"]]
        indices = range(len(queries)) if args.all_queries else [args.query_idx]
        for query_idx in indices:
            jobs.append((sha256, pred_coords, affordance_path, query_idx, queries[query_idx]))
    print(f"{len(jobs)} (object, query) pairs to evaluate")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pred_coords_list, pred_logits_list, affordance_paths, query_indices = [], [], [], []

    for start in tqdm(range(0, len(jobs), args.batch_size), desc="partial"):
        batch = jobs[start : start + args.batch_size]
        coords_list = [torch.from_numpy(coords.astype(np.int32)) for _, coords, _, _, _ in batch]
        queries = [query for *_, query in batch]
        outputs = pipeline.run_batch(coords_list, queries, **sampler_params(args))
        for (sha256, coords, affordance_path, query_idx, _), output in zip(batch, outputs):
            pred_coords_list.append(coords)
            pred_logits_list.append(output["logits"].cpu().numpy())
            affordance_paths.append(affordance_path)
            query_indices.append(query_idx)

    scores = metric(
        pred_sparse_structure_list=pred_coords_list,
        pred_logits_list=pred_logits_list,
        gt_affordance_paths=affordance_paths,
        query_indices=query_indices,
        return_per_sample=args.per_sample,
    )
    results = {
        "aiou": float(scores["aiou"]),
        "acd": float(scores["acd"]),
        "num_samples": len(jobs),
    }
    print(f"\naIoU: {results['aiou']:.4f}\naCD:  {results['acd']:.4f}  (n={results['num_samples']})")
    save_results(
        args.output or pred_dir / "affordance_partial.json",
        {
            "checkpoint": args.ckpt,
            "pred_dir": str(pred_dir),
            "data_dir": args.data_dir,
            "queries": "all" if args.all_queries else args.query_idx,
            "sampler": sampler_params(args),
            "seed": args.seed,
            "results": results,
        },
    )
    if args.per_sample:
        (pred_dir / "affordance_partial_per_sample.json").write_text(
            json.dumps(
                {
                    "sample_ids": [
                        [sha256, int(query_idx)]
                        for (sha256, _, _, query_idx, _) in jobs
                    ],
                    "aiou": [float(x) for x in scores["aiou_per_sample"]],
                    "acd": [float(x) for x in scores["acd_per_sample"]],
                }
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--task", required=True, choices=["recon", "complete", "partial"])
    parser.add_argument("--data_dir", default="trellis_data/affogato", help="Dataset directory")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Object list (sha256 per line; 'sha256,uid' for --task complete). "
        "Defaults to manifests/toys4k_test.txt for --task recon, "
        "manifests/affogato_test.txt otherwise",
    )
    parser.add_argument("--pred_dir", help="predict.py output dir (recon, partial)")
    parser.add_argument("--ckpt", default="chrockey/Affostruction", help="HF repo id or local dir")
    parser.add_argument("--annotations_dir", help="Raw Affogato annotations (complete)")
    parser.add_argument("--output", help="Result JSON path (default: next to the inputs)")
    parser.add_argument("--per_sample", action="store_true", help="Also dump per-sample scores")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--neg_cond", default="zeros", choices=["zeros", "empty_text"])
    parser.add_argument("--query_idx", type=int, default=0, help="Which query per object")
    parser.add_argument("--all_queries", action="store_true", help="partial: use all 5 queries")
    parser.add_argument("--mesh_metrics", action="store_true", help="recon: also CD + F-score")
    parser.add_argument(
        "--appearance_metrics",
        action="store_true",
        help="recon: also PSNR / LPIPS on RGB and normal renders (needs predict.py --gt_renders_dir)",
    )
    parser.add_argument("--gt_points_dir", help="GT 100k-point clouds ({sha256}.ply)")
    parser.add_argument("--gt_renders_dir", help="GT benchmark renders ({sha256}/rgb.png, normal.png)")
    parser.add_argument("--fscore_threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.manifest is None:
        default = "toys4k_test.txt" if args.task == "recon" else "affogato_test.txt"
        args.manifest = str(REPO_ROOT / "manifests" / default)

    if args.task == "recon":
        assert args.pred_dir, "--pred_dir is required for --task recon"
        assert not args.mesh_metrics or args.gt_points_dir, (
            "--gt_points_dir is required with --mesh_metrics"
        )
        assert not args.appearance_metrics or args.gt_renders_dir, (
            "--gt_renders_dir is required with --appearance_metrics"
        )
        eval_recon(args)
    elif args.task == "complete":
        assert args.annotations_dir, "--annotations_dir is required for --task complete"
        eval_complete(args)
    else:
        assert args.pred_dir, "--pred_dir is required for --task partial"
        eval_partial(args)


if __name__ == "__main__":
    main()
