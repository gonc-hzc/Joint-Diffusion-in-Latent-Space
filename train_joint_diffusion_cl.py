from omegaconf import OmegaConf
import argparse
import torch
from time import time
from typing import Any
from torchvision import transforms
from pytorch_lightning import seed_everything
import torch.utils.data as data
import pytorch_lightning as pl
from models import get_model_class, DDIMSamplerGradGuided
from dataloading import get_datasets
from os import path, environ
from pathlib import Path
import datetime
from callbacks import (
    ImageLogger,
    CUDACallback,
    SetupCallback,
    FIDScoreLogger,
    CheckpointEveryNSteps,
    PerTaskImageLogger,
)
from cl_methods.generative_replay import get_replay


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=False,
        help="path to model checkpoint file",
    )
    parser.add_argument("--task", "-t", type=int, required=True, help="task id")
    parser.add_argument("--learned", "-l", type=int, required=False, help="Learned tasks", nargs="+")
    parser.add_argument("--new", "-n", type=Path, required=False, help="Ckpt to new task data generator")
    parser.add_argument("--old", "-o", type=Path, required=False, help="Ckpt to old tasks data generator")
    parser.add_argument("--tags", type=str, required=False, help="Additional tags", nargs="+")
    parser.add_argument("--dir", "-d", type=str, required=False, help="Name for experiments log dir")
    parser.add_argument("--seed", "-s", type=int, required=False, help="Seed")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Override train and validation batch size for controlled comparisons",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        required=False,
        help="Number of physical batches per optimizer step",
    )
    parser.add_argument(
        "--replay-sample-batch-size",
        type=int,
        required=False,
        help="Generation batch size used while building the replay dataset",
    )
    parser.add_argument(
        "--saved-samples",
        type=Path,
        required=False,
        help="Use an existing replay image tensor instead of sampling new images",
    )
    parser.add_argument(
        "--saved-labels",
        type=Path,
        required=False,
        help="Labels corresponding to --saved-samples",
    )
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    old_generator_path = str(args.old) if args.old is not None else None
    new_generator_path = str(args.new) if args.new is not None else None
    current_task = args.task
    tasks_learned = args.learned if args.learned is not None else []
    tags = args.tags if args.tags is not None else []
    custom_dir = args.dir
    seed = args.seed if args.seed is not None else 42
    if args.replay_sample_batch_size is not None and args.replay_sample_batch_size < 1:
        parser.error("--replay-sample-batch-size must be at least 1")
    if (args.saved_samples is None) != (args.saved_labels is None):
        parser.error("--saved-samples and --saved-labels must be provided together")
    if args.saved_samples is not None:
        if not args.saved_samples.is_file():
            parser.error(f"saved samples file does not exist: {args.saved_samples}")
        if not args.saved_labels.is_file():
            parser.error(f"saved labels file does not exist: {args.saved_labels}")

    config = OmegaConf.load(config_path)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        config.dataloaders.validation.batch_size = args.batch_size
        for train_group in config.dataloaders.train:
            for dataset_config in train_group.cl_split.datasets:
                dataset_config.dataset.batch_size = args.batch_size
    seed_everything(seed)

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    if args.accumulate_grad_batches is not None:
        if args.accumulate_grad_batches < 1:
            parser.error("--accumulate-grad-batches must be at least 1")
        trainer_config["accumulate_grad_batches"] = args.accumulate_grad_batches
        validation_period = trainer_config.get("check_val_every_n_epoch", 1)
        trainer_config["check_val_every_n_epoch"] = validation_period * args.accumulate_grad_batches
    trainer_config["devices"] = -1
    trainer_opt = trainer_config
    lightning_config.trainer = trainer_config

    dl_config_orig = config.pop("dataloaders")
    dl_config = OmegaConf.to_container(dl_config_orig, resolve=True)
    tasks_datasets, tasks_bs, test_ds, test_bs, tasks = get_datasets(dl_config)

    test_dl = data.DataLoader(
        test_ds,
        test_bs,
        shuffle=False,
        num_workers=16,
    )

    cl_config = config.pop("cl")
    if args.saved_samples is not None:
        cl_config["saved_samples"] = str(args.saved_samples)
        cl_config["saved_labels"] = str(args.saved_labels)

    replay_sample_batch_size = (
        args.replay_sample_batch_size
        if args.replay_sample_batch_size is not None
        else cl_config.get("sample_batch_size", 2000)
    )
    reply_buff = get_replay(cl_config.get("reply_type"))(
        train_bs=tasks_bs,
        sample_bs=replay_sample_batch_size,
        dl_num_workers=8,
    )

    model_type = config.model.get("model_type")
    params = config.model.get("params", dict())
    new_generator = None
    if new_generator_path is not None:
        if model_type in [
            "joint_diffusion_knowledge_distillation",
            "joint_diffusion_adversarial_knowledge_distillation",
        ]:
            params["new_model"] = None
            params["old_model"] = None
        config.model.params["ckpt_path"] = new_generator_path
        new_generator = get_model_class(model_type)(**params)
    old_generator = None
    if old_generator_path is not None:
        if model_type in [
            "joint_diffusion_knowledge_distillation",
            "joint_diffusion_adversarial_knowledge_distillation",
        ]:
            params["new_model"] = None
            params["old_model"] = None
        config.model.params["ckpt_path"] = old_generator_path
        old_generator = get_model_class(model_type)(**params)
    if model_type in ["joint_diffusion_knowledge_distillation", "joint_diffusion_adversarial_knowledge_distillation"]:
        params = OmegaConf.to_container(params, resolve=True)
        params["new_model"] = new_generator
        params["old_model"] = old_generator

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    model = get_model_class(model_type)(**params)

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now if custom_dir is None else custom_dir
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs: dict[str, Any] = dict()
    try:
        per_class = f'{dl_config["train"][0]["cl_split"]["datasets"][0]["ssl_split"]["num_labeled"]} per class'
    except Exception:
        per_class = "all labels"
    tags.extend(
        [
            dl_config["validation"]["name"],
            per_class,
            config.model.get("model_type"),
            f"task {current_task}",
            f"learned tasks {tasks_learned}",
            f"physical batch {tasks_bs[0]}",
            f"gradient accumulation {trainer_config.get('accumulate_grad_batches', 1)}",
            f"replay sample batch {replay_sample_batch_size}",
        ]
    )
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(
        name=nowname, id=nowname, tags=tags, project="Joint-Diffusion-in-Latent-Space"
    )

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
            "mode": "max",
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 1

    callback_cfg = lightning_config.get("callbacks", OmegaConf.create())
    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg["params"]),
        SetupCallback(
            resume=False,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config,
            dl_config=dl_config_orig,
        ),
        CUDACallback(),
        CheckpointEveryNSteps(10000, prefix="ckpt_"),
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(PerTaskImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        fid_cfg = dict(fid_cfg)
        fid_cfg["real_dl"] = test_dl
        fid_cfg["device"] = torch.device("cuda")
        trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    def get_generator(generator):
        def generate_samples(batch, labels):
            generator.sampling_method = cl_config["sampling_method"]
            generator.sample_grad_scale = cl_config["grad_scale"]
            soft_labels = cl_config.get("use_soft_labels", False)
            ddim = cl_config.get("ddim_steps", False)
            ema = cl_config.get("use_ema", True)
            with torch.no_grad():
                if labels is not None:
                    labels = torch.tensor(labels, device=generator.device)
                    generator.sample_classes = labels
                if not ddim:
                    if ema:
                        with generator.ema_scope():
                            samples = generator.sample(batch_size=batch)
                    else:
                        samples = generator.sample(batch_size=batch)
                else:
                    shape = (
                        generator.channels,
                        generator.image_size,
                        generator.image_size,
                    )
                    if ema:
                        # so that ema works
                        with generator.ema_scope():
                            ddim_sampler = DDIMSamplerGradGuided(generator)
                            samples, _ = ddim_sampler.sample(
                                S=ddim,
                                batch_size=batch,
                                shape=shape,
                                cond=None,
                                verbose=False,
                            )
                    else:
                        ddim_sampler = DDIMSamplerGradGuided(generator)
                        samples, _ = ddim_sampler.sample(
                            S=ddim,
                            batch_size=batch,
                            shape=shape,
                            cond=None,
                            verbose=False,
                        )

                unet = generator.model.diffusion_model
                representations = unet.just_representations(
                    samples,
                    torch.zeros(samples.shape[0], device=samples.device),
                    context=None,
                    pooled=False,
                )
                pooled_representations = generator.transform_representations(representations)
                pred = generator.classifier(pooled_representations)
                pred_labels = pred.argmax(dim=-1)
                if labels is not None:
                    samples = samples[pred_labels == labels]
                    labels = labels[pred_labels == labels] if not soft_labels else pred[pred_labels == labels]
                else:
                    labels = pred_labels if not soft_labels else pred

                samples = samples.cpu()
                mean = cl_config["mean"]
                std = cl_config["std"]
                denormalize = transforms.Compose(
                    [
                        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                        transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                    ]
                )
                samples = denormalize(samples)
                samples = torch.clamp(samples, 0, 1)
            generator.sampling_method = "unconditional"
            generator.sample_classes = None
            return samples, labels.cpu()

        return generate_samples

    generate_old_samples = get_generator(old_generator)

    new_samples_generate = None
    if new_generator is not None:
        new_samples_generate = get_generator(new_generator)

    prev_tasks = []
    for i, task in enumerate(tasks):
        if i in tasks_learned:
            prev_tasks.extend(task)
    for i, (datasets, task) in enumerate(zip(tasks_datasets, tasks)):
        if i == current_task:
            if old_generator is not None:
                old_generator.to(torch.device("cuda"))
            if new_generator is not None:
                new_generator.to(torch.device("cuda"))
            (labeled_ds, unlabeled_ds) = datasets if len(datasets) == 2 else (datasets[0], None)
            start_sampling_time = time()
            train_dls = reply_buff.get_data_for_task(
                sup_ds=labeled_ds,
                unsup_ds=unlabeled_ds,
                prev_tasks=prev_tasks,
                samples_per_task=cl_config["samples_per_class"],
                old_sample_generator=generate_old_samples,
                new_sample_generator=(
                    new_samples_generate if unlabeled_ds is not None or new_generator is not None else True
                ),  # dummy for class conditioned baseline stuff
                current_task=task,
                filename=nowname,
                saved_samples=cl_config.get("saved_samples", None),
                saved_labels=cl_config.get("saved_labels", None),
            )
            print(f"SAMPLING TIME: {time() - start_sampling_time} s")
            if new_generator is not None:
                del new_generator
            if old_generator != model:
                del old_generator

            trainer = pl.Trainer(**trainer_opt, **trainer_kwargs)

            weight_reinit = cl_config.get("weight_reinit", "none")
            if weight_reinit == "none":
                pass
            elif weight_reinit == ["unused classes", "freeze+reinit_unused"]:
                torch.nn.init.xavier_uniform_(model.classifier[-1].weight[len(prev_tasks) :])
            elif weight_reinit == "classifier":
                for layer in model.classifier:
                    if hasattr(layer, "weight"):
                        torch.nn.init.xavier_uniform_(layer.weight)
            elif weight_reinit in ["freeze", "freeze+reinit_unused"]:

                def zero_grad_old_tasks(grad):
                    grad_clone = grad.clone()
                    grad_clone[: len(prev_tasks)] = 0
                    return grad_clone

                model.classifier[-1].weight.register_hook(zero_grad_old_tasks)
                model.classifier[-1].bias.register_hook(zero_grad_old_tasks)
            trainer.fit(
                model,
                train_dataloaders=train_dls,
                val_dataloaders=test_dl,
            )
