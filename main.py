import os
import shutil
import argparse
from easydict import EasyDict as edict
import shutil
import yaml
import random
import time
import datetime
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

from utils import create_logger, create_optimizer, create_scheduler, auto_resume_helper
from data.dataset import Dataset
from model.physgm import PhysGM
from torch.utils.tensorboard import SummaryWriter


def main():
    # config setup
    parser = argparse.ArgumentParser(description="PhysGM arguments")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--default-config", type=str, help="path to default config file"
    )
    parser.add_argument("--evaluation", action="store_true", help="evaluation mode")
    args = parser.parse_args()
    config_name = os.path.basename(args.config).split(".")[0]
    config = yaml.safe_load(open(args.config, "r"))

    def recursive_merge(dict1, dict2):
        for key, value in dict2.items():
            if key not in dict1:
                dict1[key] = value
            elif isinstance(value, dict):
                dict1[key] = recursive_merge(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    if args.default_config is not None:
        default_config = yaml.safe_load(open(args.default_config, "r"))
        default_config = recursive_merge(default_config, config)
        config = default_config

    config_dict = config
    config = edict(config)

    if args.evaluation:
        config.evaluation = True
    if config.get("config_name", None) is not None:
        config_name = config.config_name

    if config.get("auto_resume", True):
        checkpoint_dir = os.path.join(config.checkpoint_dir, config_name)
    else:
        timestamp = datetime.datetime.now().strftime("_%m%d_%H%M%S")
        checkpoint_dir = os.path.join(config.checkpoint_dir, config_name + timestamp)

    backup_dir = os.path.join(checkpoint_dir, "backup")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(backup_dir, "config.yaml"))

    # torch and DDP setup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    device = "cuda:{}".format(local_rank)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    seed = 1111 + rank
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = config.use_tf32
    torch.backends.cudnn.allow_tf32 = config.use_tf32

    logger_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(logger_dir, exist_ok=True)
    logger = create_logger(output_dir=logger_dir, dist_rank=rank, name=config_name)
    logger.info(
        f"Rank {rank} / {world_size} with local rank {local_rank} / {local_world_size} and group rank {group_rank}"
    )
    logger.info("Config:\n" + yaml.dump(config_dict, sort_keys=False))

    torch.distributed.barrier()

    # dataloader
    dataset = Dataset(config)
    if rank == 0:
        data_example = dataset[0]
        os.makedirs(os.path.join(checkpoint_dir, "data_example"), exist_ok=True)
        data_desc = ""
        for key, value in data_example.items():
            if isinstance(value, torch.Tensor):
                data_desc += "data key: {}, shape: {}\n".format(key, value.size())
            else:
                data_desc += "data key: {}, value: {}\n".format(key, value)
            if key == "input_images":
                input_images = value  # (V, C, H, W)
                input_images = input_images.permute(1, 2, 0, 3).flatten(
                    2, 3
                )  # (C, H, V*W)
                torchvision.utils.save_image(
                    input_images,
                    os.path.join(checkpoint_dir, "data_example", "input_images.png"),
                )
        logger.info("Data example:\n" + data_desc)
        tensorboard_dir = os.path.join(checkpoint_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    torch.distributed.barrier()

    datasampler = DistributedSampler(
        dataset, shuffle=not config.get("evaluation", False)
    )
    batch_size_per_gpu = config.training.batch_size_per_gpu
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config.training.prefetch_factor,
        sampler=datasampler,
    )

    # model setup
    model = PhysGM(config, device).to(device)
   

    if rank == 0:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"=== Frozen Mode: Only training {num_trainable / 1e6:.4f}M parameters ==="
        )

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    enable_grad_scaler = config.use_amp and config.amp_dtype == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
    amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}

    # optimizer, scheduler, load checkpoint
    if not config.get("evaluation", False):
        train_steps = config.training.train_steps
        grad_accum_steps = config.training.grad_accum_steps
        param_update_steps = train_steps
        train_steps = train_steps * grad_accum_steps
        total_batch_size = batch_size_per_gpu * world_size * grad_accum_steps
        num_epochs = int(param_update_steps * total_batch_size / len(dataset))
        logger.info(
            f"train_steps: {train_steps}, grad_accum_steps: {grad_accum_steps}, param_update_steps: {param_update_steps}, batch_size_per_gpu: {batch_size_per_gpu}, world_size: {world_size}, batch_size_total: {total_batch_size}, dataset_size: {len(dataset)}, num_epochs: {num_epochs}"
        )
        optimizer = create_optimizer(
            model,
            config.training.weight_decay,
            config.training.lr,
            (config.training.beta1, config.training.beta2),
        )
        scheduler = create_scheduler(
            optimizer,
            param_update_steps,
            config.training.warmup_steps,
            config.training.get("scheduler_type", "cosine"),
        )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Number of parameters: {num_params / 1e6:.2f}M, trainable: {num_trainable_params / 1e6:.2f}M"
    )
    train_steps_done = 0
    resume_file = auto_resume_helper(checkpoint_dir)
    auto_resume = resume_file is not None

    if resume_file is None:
        resume_file = config.training.get("resume_ckpt", None)

    if resume_file is None:
        logger.info("No checkpoint found, starting from scratch.")
    else:
        logger.info(f"Attempting to resume from checkpoint: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device)
        logger.info(
            f"Checkpoint file content reports train_steps_done: {checkpoint.get('train_steps_done', 'NOT FOUND')}"
        )

        if isinstance(model, DDP):
            status = model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            status = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Loaded model weights with status: {status}")


    logger.info(f"FINAL train_steps_done before training loop is: {train_steps_done}")



    torch.distributed.barrier()
    model.train()

    local_step = 0
    warmup_steps_finetune = config.training.warmup_steps

    
    

    len_dataset = len(dataset) // batch_size_per_gpu * batch_size_per_gpu
    cur_epoch = train_steps_done * total_batch_size // len_dataset // grad_accum_steps
    datasampler.set_epoch(cur_epoch)
    dataloader_iter = iter(dataloader)
    param_optim_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    param_optim_list = [p for p in param_optim_dict.values()]
    train_steps_start = train_steps_done
    param_update_steps_start = train_steps_done // grad_accum_steps

    while train_steps_done <= train_steps:
        param_update_steps_done = train_steps_done // grad_accum_steps

        try:
            data = next(dataloader_iter)
        except StopIteration:
            logger.info("We have exhausted the dataloader iterator, resetting it")
            cur_epoch = param_update_steps_done * total_batch_size // len_dataset
            datasampler.set_epoch(cur_epoch)
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        if local_step == warmup_steps_finetune:
            if rank == 0:
                logger.info(
                    f"=== Local Step {local_step}: Unfreezing backbone for fine-tuning ==="
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = config.training.lr * 0.1

        local_step += 1

        update_param = (train_steps_done + 1) % grad_accum_steps == 0
        context = torch.autocast(
            enabled=config.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.amp_dtype],
        )
        if update_param:
            with context:
                ret_dict = model(data)
        else:
            with model.no_sync(), context:
                ret_dict = model(data)

        loss_dict = ret_dict["loss"]
        total_loss = loss_dict["total_loss"]

        scaler.scale(total_loss / grad_accum_steps).backward()
        train_steps_done += 1
        param_update_steps_done = train_steps_done // grad_accum_steps

        skip_optimizer_step = False
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            loss_dict["total_loss"] = torch.tensor(0.0).to(device)

        if update_param and (not skip_optimizer_step):
            scaler.unscale_(optimizer)
            with torch.no_grad():
                for n, p in param_optim_dict.items():
                    if p.grad is None:
                        logger.warning(
                            f"step {train_steps_done} found a None grad for {n}"
                        )
                    else:
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-3, neginf=-1e-3)

            total_grad_norm = 0.0

            grad_clip_norm = 30.0

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                param_optim_list, max_norm=grad_clip_norm
            ).item()

            if total_grad_norm > grad_clip_norm:
                logger.warning(
                    f"step {train_steps_done} grad norm {total_grad_norm} was clipped to {grad_clip_norm}"
                )

            if not skip_optimizer_step:
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # logging and checkpointing
        if rank == 0 and update_param:
            gaussian_usage = ret_dict["gaussian_usage"].mean().item()
            loss_dict_items = {k: v.item() for k, v in loss_dict.items()}
            for k, v in loss_dict_items.items():
                writer.add_scalar(f"Loss/{k}", v, param_update_steps_done)

            writer.add_scalar(
                "Metrics/gaussian_usage", gaussian_usage, param_update_steps_done
            )
            writer.add_scalar(
                "Metrics/learning_rate",
                scheduler.get_last_lr()[0],
                param_update_steps_done,
            )
            writer.add_scalar(
                "Metrics/total_grad_norm", total_grad_norm, param_update_steps_done
            )

            if "psnr" in loss_dict_items:
                writer.add_scalar(
                    "Metrics/psnr", loss_dict_items["psnr"], param_update_steps_done
                )

            if (
                param_update_steps_done % config.training.print_every == 0
                or param_update_steps_done < param_update_steps_start + 100
            ):
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                loss_str += f", gaussian_usage: {gaussian_usage:.4f}"
                memory_usage = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                logger.info(
                    f"\nStep {param_update_steps_done} / {train_steps // grad_accum_steps}, Epoch {cur_epoch}\n{loss_str}, scene_scale: {ret_dict['scene_scale']:.2f}, memory: {memory_usage:.2f} MB"
                )

            if (
                param_update_steps_done % config.training.checkpoint_every == 0
                or train_steps_done == train_steps
            ):
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_steps_done": train_steps_done,
                }
                ckpt_filename = f"checkpoint_{param_update_steps_done:06d}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                torch.save(checkpoint, ckpt_path)
                last_path = os.path.join(checkpoint_dir, "last.pt")
                if os.path.islink(last_path) or os.path.exists(last_path):
                    os.remove(last_path)
                os.symlink(ckpt_filename, last_path)
                logger.info(f"Saved checkpoint at step {param_update_steps_done}")

            if (
                param_update_steps_done % config.training.vis_every == 0
                or param_update_steps_done < param_update_steps_start + 20
            ):
                save_gaussian = (
                    param_update_steps_done % config.training.save_gaussian_every == 0
                    or param_update_steps_done == 1
                )
                model.module.save_visualization(
                    data,
                    ret_dict,
                    os.path.join(checkpoint_dir, f"vis_{param_update_steps_done:06d}"),
                    save_gaussian=save_gaussian,
                    save_video=False,
                )

        torch.distributed.barrier()

    if rank == 0:
        writer.close()

    torch.distributed.barrier()
    destroy_process_group()


if __name__ == "__main__":
    main()
