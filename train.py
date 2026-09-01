import os
import sys
import json
import glob
import argparse
import traceback
from datetime import timedelta
from easydict import EasyDict as edict
import warnings

os.environ.setdefault("SPCONV_ALGO", "native")

warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import torch.distributed as dist

torch.set_float32_matmul_precision("high")

from affostruction import models, datasets, trainers


def find_ckpt(cfg):
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if len(files) != 0:
                cfg.load_ckpt = max(
                    [int(os.path.basename(f).split("step")[-1].split(".")[0]) for f in files]
                )
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += (
            f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        )
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def main(cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=4)
        )

    dataset_args = (
        cfg.dataset.args.copy() if hasattr(cfg.dataset.args, "copy") else dict(cfg.dataset.args)
    )
    if "split" not in dataset_args:
        dataset_args["split"] = "train"
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **dataset_args)

    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f"\n\nBackbone: {name}\n" + model_summary)
            with open(os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w") as fp:
                print(model_summary, file=fp)

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )

    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--load_dir", type=str, default="", help="Load directory, default to output_dir"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint step to resume training, default to latest",
    )
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory")
    parser.add_argument("--auto_retry", type=int, default=3, help="Number of retries on error")
    parser.add_argument("--tryrun", action="store_true", help="Try run without training")
    parser.add_argument("--profile", action="store_true", help="Profile training")
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    config = json.load(open(opt.config))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n\nConfig:")
        print("=" * 80)
        print(json.dumps(cfg.__dict__, indent=4))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
            print(" ".join(["python"] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    attempts = max(1, cfg.auto_retry)
    for attempt in range(attempts):
        try:
            cfg = find_ckpt(cfg)
            main(cfg)
            break
        except Exception:
            traceback.print_exc()
            if attempt + 1 == attempts:
                raise
            print(f"Retrying ({attempt + 1}/{attempts})...")
