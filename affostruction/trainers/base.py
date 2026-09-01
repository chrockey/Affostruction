from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils

try:
    import wandb
except ImportError:
    wandb = None

from .fp16 import *
from ..utils.general import *
from ..utils.data import recursive_to_device, cycle, ResumableSampler


class Trainer:
    """
    Base class for training.
    """

    def __init__(
        self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode="inflat_all",
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        reset_step=False,
        log_param_stats=False,
        prefetch_data=False,
        save_best_only=True,
        enable_visualization=False,
        wandb_sample_count=8,
        save_samples_to_disk=False,
        validation_config={},
        num_workers=4,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs,
    ):
        assert (
            batch_size is not None or batch_size_per_gpu is not None
        ), "Either batch_size or batch_size_per_gpu must be specified."

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.reset_step = reset_step
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.save_best_only = save_best_only
        self.enable_visualization = enable_visualization
        self.wandb_sample_count = wandb_sample_count
        self.save_samples_to_disk = save_samples_to_disk
        self.validation_config = validation_config
        self.num_workers = num_workers
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck

        assert (
            i_save >= i_log
        ), f"i_save ({i_save}) must be >= i_log ({i_log}) for proper averaged loss calculation"

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = (
            batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        )
        self.batch_size_per_gpu = (
            batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        )
        assert (
            self.batch_size % self.world_size == 0
        ), "Batch size must be divisible by the number of GPUs."
        assert (
            self.batch_size_per_gpu % self.batch_split == 0
        ), "Batch size per GPU must be divisible by batch split."

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)

        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "ckpts"), exist_ok=True)
            if self.save_samples_to_disk:
                os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            if self.validation_config.get("enabled", False) and not hasattr(
                self, "val_metric_best"
            ):
                self.val_metric_best = (
                    float("-inf")
                    if self.validation_config["metric_mode"] == "max"
                    else float("inf")
                )
            if wandb is not None and os.environ.get("WANDB_API_KEY"):
                run_name = os.path.basename(self.output_dir.rstrip("/"))
                wandb_tags = []
                if "SLURM_JOBID" in os.environ:
                    wandb_tags.append(os.environ["SLURM_JOBID"])
                if "NODE" in os.environ:
                    wandb_tags.append(os.environ["NODE"])
                self.wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "affostruction"),
                    name=run_name,
                    dir=self.output_dir,
                    resume="allow",
                    tags=wandb_tags,
                )
            else:
                self.wandb_run = None
                print("WandB logging disabled (install wandb and set WANDB_API_KEY to enable).")

        if self.validation_config.get("enabled", False):
            self._setup_validation(**kwargs)

        if self.world_size > 1:
            self.check_ddp()

        if self.is_master:
            print("\n\nTrainer initialized.")
            print(self)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, "device"):
                return model.device
        return next(list(self.models.values())[0].parameters()).device

    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )

        cpu_per_gpu = int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        if self.num_workers is None:
            num_workers = cpu_per_gpu
        else:
            num_workers = min(cpu_per_gpu, self.num_workers)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            sampler=self.data_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        self.data_iterator = cycle(self.dataloader)

    def _get_dataset_init_kwargs(self):
        """Extract initialization kwargs from training dataset to create validation dataset."""
        kwargs = {}

        if hasattr(self.dataset, "latent_model"):
            kwargs["latent_model"] = self.dataset.latent_model
        if hasattr(self.dataset, "min_aesthetic_score"):
            kwargs["min_aesthetic_score"] = self.dataset.min_aesthetic_score
        if hasattr(self.dataset, "normalization"):
            kwargs["normalization"] = self.dataset.normalization
        if hasattr(self.dataset, "image_size"):
            kwargs["image_size"] = self.dataset.image_size

        return kwargs

    def _setup_validation(self, **kwargs):
        """
        Setup validation dataset and dataloader.
        Called during initialization if validation is enabled (master rank only for single GPU validation).
        """
        if not self.is_master:
            return

        if self.is_master:
            print(f"\nSetting up validation from {self.validation_config['data_dir']}...")

        val_dataset_kwargs = self._get_dataset_init_kwargs()
        val_dataset_kwargs["roots"] = self.validation_config["data_dir"]
        if self.validation_config.get("manifest"):
            val_dataset_kwargs["manifest"] = self.validation_config["manifest"]
            val_dataset_kwargs["split"] = None
        else:
            val_dataset_kwargs["split"] = "test"

        if "num_views" in self.validation_config:
            val_dataset_kwargs["num_views"] = self.validation_config["num_views"]

        val_dataset_kwargs.update(kwargs)

        val_dataset_class = self.dataset.__class__
        try:
            self.val_dataset = val_dataset_class(**val_dataset_kwargs)
        except Exception as e:
            print(f"Warning: Failed to create validation dataset: {e}")
            print("Disabling validation-based checkpoint saving")
            self.validation_config = {}
            return

        if len(self.val_dataset) == 0:
            print("Warning: No validation samples found")
            print("Disabling validation-based checkpoint saving")
            self.validation_config = {}
            return

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.validation_config.get("batch_size", 4),
            shuffle=False,
            sampler=None,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn
            if hasattr(self.val_dataset, "collate_fn")
            else None,
        )

        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Validation batches: {len(self.val_dataloader)} (single GPU validation)")

    @torch.no_grad()
    def run_validation(self):
        """
        Run validation.
        Subclasses should override this method to implement validation logic.
        """
        pass

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self, current_val_metric=None):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass

    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f"dataset_{k}", v) for k, v in vis.items()]
        else:
            save_cfg = [("dataset", vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=None, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if num_samples is None:
            num_samples = self.wandb_sample_count

        if self.is_master:
            print(f"\nSampling {num_samples} images...", end="")

        if suffix is None:
            suffix = f"step{self.step:07d}"

        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(
            num_samples_per_process, batch_size=batch_size, verbose=verbose
        )

        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}

        if self.world_size > 1:
            for key in samples.keys():
                samples[key]["value"] = samples[key]["value"].contiguous()
                if self.is_master:
                    all_images = [
                        torch.empty_like(samples[key]["value"]) for _ in range(self.world_size)
                    ]
                else:
                    all_images = []
                dist.gather(samples[key]["value"], all_images, dst=0)
                if self.is_master:
                    samples[key]["value"] = torch.cat(all_images, dim=0)[:num_samples]

        if self.is_master:
            wandb_images = {}

            if self.save_samples_to_disk:
                os.makedirs(os.path.join(self.output_dir, "samples", suffix), exist_ok=True)

            for key in samples.keys():
                if samples[key]["type"] == "image":
                    grid_image = utils.make_grid(
                        samples[key]["value"],
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )

                    if self.save_samples_to_disk:
                        utils.save_image(
                            samples[key]["value"],
                            os.path.join(
                                self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                            ),
                            nrow=int(np.sqrt(num_samples)),
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )

                    if self.wandb_run is not None:
                        wandb_images[f"samples/{key}"] = wandb.Image(
                            grid_image, caption=f"{key} at step {self.step}"
                        )

                elif samples[key]["type"] == "number":
                    min = samples[key]["value"].min()
                    max = samples[key]["value"].max()
                    images = (samples[key]["value"] - min) / (max - min)
                    grid_image = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )

                    if self.save_samples_to_disk:
                        save_image_with_notes(
                            grid_image,
                            os.path.join(
                                self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                            ),
                            notes=f"{key} min: {min}, max: {max}",
                        )

                    if self.wandb_run is not None:
                        wandb_images[f"samples/{key}"] = wandb.Image(
                            grid_image,
                            caption=f"{key} at step {self.step} (min: {min:.4f}, max: {max:.4f})",
                        )

            if wandb_images and self.wandb_run is not None:
                wandb.log(wandb_images, step=self.step)

        if self.is_master:
            print(" Done.")

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(self, **mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(
                    next(self.data_iterator), self.device, non_blocking=True
                )
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(
                next(self.data_iterator), self.device, non_blocking=True
            )
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)

        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {
                        k: v[
                            i
                            * batch_size
                            // self.batch_split : (i + 1)
                            * batch_size
                            // self.batch_split
                        ]
                        for k, v in data.items()
                    }
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError("Data must be a dict or a list of dicts.")

        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print("\nStarting training...")
            if self.save_samples_to_disk:
                self.snapshot_dataset()
        if self.enable_visualization:
            if self.step == 0:
                self.snapshot(suffix="init")
            else:
                self.snapshot(suffix=f"resume_step{self.step:07d}")

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            if self.is_master and self.step % self.i_print == 0:
                time_diff = time_elapsed - time_last_print
                speed = self.i_print / time_diff * 3600 if time_diff > 0 else float("inf")
                eta_str = f"{(self.max_steps - self.step) / speed:.2f} h" if speed > 0 else "N/A"
                columns = [
                    f"Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {eta_str}",
                ]
                print(" | ".join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            if (
                self.world_size > 1
                and self.i_ddpcheck is not None
                and self.step % self.i_ddpcheck == 0
            ):
                self.check_ddp()

            if self.enable_visualization and self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

                log[-1][1]["time"] = {
                    "step": time_end - time_start,
                    "elapsed": time_elapsed,
                }

                if step_log is not None:
                    log[-1][1].update(step_log)

                if self.fp16_mode == "amp":
                    log[-1][1]["scale"] = self.scaler.get_scale()
                elif self.fp16_mode == "inflat_all":
                    log[-1][1]["log_scale"] = self.log_scale

                if self.step % self.i_log == 0:
                    log_str = "\n".join(
                        [
                            f"{step}: {json.dumps(ensure_json_serializable(log))}"
                            for step, log in log
                        ]
                    )
                    with open(os.path.join(self.output_dir, "log.txt"), "a") as log_file:
                        log_file.write(log_str + "\n")

                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep="/")
                    if self.wandb_run is not None:
                        wandb.log(log_show, step=self.step)

                    log = []

                if self.step % self.i_save == 0:
                    val_metric = None
                    metric_value = None

                    if self.validation_config.get("enabled", False) and self.is_master:
                        val_metric = self.run_validation()

                        if val_metric is not None:
                            if self.wandb_run is not None:
                                wandb.log(
                                    {f"metric/val_{k}": v for k, v in val_metric.items()},
                                    step=self.step,
                                )

                            assert (
                                self.validation_config["primary_metric"] in val_metric
                            ), f"Primary metric {self.validation_config['primary_metric']} not found in validation metrics"
                            metric_value = val_metric.get(self.validation_config["primary_metric"])

                    if self.is_master:
                        self.save(metric_value)

        if self.enable_visualization:
            self.snapshot(suffix="final")

        if self.is_master:
            if self.wandb_run is not None:
                wandb.finish()
            print("Training finished.")

        if self.world_size > 1:
            dist.destroy_process_group()

    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(self.output_dir, "profile")
            ),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                data_list = self.load_data()
                self.run_step(data_list)
                prof.step()
