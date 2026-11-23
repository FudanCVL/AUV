# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
import random
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from hydra import compose
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

# Allow running this file directly from any working directory.
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.utils.train_utils import makedir, register_omegaconf_resolvers


os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Run training in one GPU process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Spawned processes need their own OmegaConf resolver registration.
    if not OmegaConf.has_resolver("get_method"):
        register_omegaconf_resolvers()

    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def single_node_runner(cfg, main_port: int):
    if cfg.launcher.num_nodes != 1:
        raise ValueError("Local training supports one node only")

    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        torch.multiprocessing.start_processes(
            single_proc_run,
            args=(main_port, cfg, num_proc),
            nprocs=num_proc,
            start_method="spawn",
        )


def validate_stages(cfg):
    stage1 = OmegaConf.merge(cfg, {"stage": "stage1"})
    stage2 = OmegaConf.merge(cfg, {"stage": "stage2"})
    for key in ("launcher.experiment_log_dir", "trainer.checkpoint.save_dir"):
        first = Path(OmegaConf.select(stage1, key)).resolve()
        second = Path(OmegaConf.select(stage2, key)).resolve()
        if first == second:
            raise ValueError(
                f"Stage 1 and Stage 2 must use different directories: {key}"
            )
    if (
        stage1.trainer.data.train.datasets[0].dataset.datasets[0].sampler.num_frames != 1
        or stage2.trainer.data.train.datasets[0].dataset.datasets[0].sampler.num_frames != 8
    ):
        raise ValueError("Stage 1 requires one frame; Stage 2 requires eight frames")
    if stage1.trainer.max_epochs < 1 or stage2.trainer.max_epochs < 1:
        raise ValueError("Each stage must have at least one epoch")


def prepare_stage_resume(cfg):
    """Resume legacy training state or warm-start from a compact UAP checkpoint."""
    path = Path(cfg.trainer.checkpoint.save_dir) / "checkpoint.pt"
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "uap" in checkpoint and "model" not in checkpoint:
        cfg.trainer.model.pretrained_uap = None
        print(
            f"{cfg.stage}: loading UAP from {path}; restarting at epoch 0 with a "
            "fresh optimizer (no training state saved).",
            flush=True,
        )
        return False
    epoch = int(checkpoint["epoch"])
    if epoch >= cfg.trainer.max_epochs:
        print(f"{cfg.stage}: already completed {epoch} epochs; skipping.", flush=True)
        return True
    cfg.trainer.model.pretrained_uap = None
    print(f"{cfg.stage}: resuming from epoch {epoch} in {path}", flush=True)
    return False


def main(args) -> None:
    cfg = compose(config_name=args.config, overrides=args.overrides)
    if cfg.get("stages") is not None:
        validate_stages(cfg)
        if args.stage == "all":
            # Separate processes release each stage's CUDA memory and distributed state.
            for stage in ("stage1", "stage2"):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "-c",
                    args.config,
                    "--stage",
                    stage,
                ]
                for flag, value in (
                    ("--num-gpus", args.num_gpus),
                    ("--num-nodes", args.num_nodes),
                    ("--use-cluster", args.use_cluster),
                ):
                    if value is not None:
                        command.extend([flag, str(value)])
                if args.dry_run:
                    command.append("--dry-run")
                command.extend(args.overrides)
                subprocess.run(command, check=True)
            return
        cfg.stage = args.stage
    elif args.stage != "all":
        raise ValueError("--stage requires a config with a stages section")

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )

    submitit_conf = cfg.get("submitit")
    cluster_requested = args.use_cluster
    if cluster_requested is None and submitit_conf is not None:
        cluster_requested = submitit_conf.get("use_cluster", False)
    if cluster_requested:
        raise NotImplementedError(
            "Cluster submission is not implemented in this entry point"
        )

    if args.num_nodes not in (None, 1):
        raise ValueError("Local training supports one node only")

    cfg.launcher.num_nodes = 1
    if args.num_gpus is not None:
        if args.num_gpus < 1:
            raise ValueError("--num-gpus must be at least 1")
        cfg.launcher.gpus_per_node = args.num_gpus

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    if cfg.get("stages") is not None and prepare_stage_resume(cfg):
        return

    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Experiment Log Dir:\n{cfg.launcher.experiment_log_dir}")
    main_port = random.randint(10000, 29999)
    print(f"Using port {main_port} with {cfg.launcher.gpus_per_node} GPU(s)")
    single_node_runner(cfg, main_port)


if __name__ == "__main__":
    import sam2  # Initializes the local SAM2 Hydra config module.

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="configs/attack.yaml", help="Training config name"
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        choices=(0, 1),
        default=None,
        help="Cluster submission is not implemented in this entry point",
    )
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--stage", choices=("all", "stage1", "stage2"), default="all")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configs without training"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides: key=value")
    args = parser.parse_args()
    register_omegaconf_resolvers()
    main(args)
