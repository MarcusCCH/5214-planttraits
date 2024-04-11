import argparse
from random import shuffle, seed
import os
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from training import *

from utils.data import get_training_data, CaseCollator
from optimisers import (
    get_optimizer,
    lr_cosine_policy,
    lr_linear_policy,
    lr_step_policy,
)

from models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
)


def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
        ]
    }
    return models





def add_parser_arguments(parser):
    model_names = available_models().keys()

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='data/')

    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    parser.add_argument('--case_name', type=str, default="test")
    
    parser.add_argument(
        "--archs",
        "-a",
        default="efficientnet-b0",
        choices=model_names,
        help="model architecture: "
        + " | ".join(model_names)
        + " (default: efficientnet-b0)",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--prefetch",
        default=2,
        type=int,
        metavar="N",
        help="number of samples prefetched by each loader",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=-1,
        type=int,
        metavar="N",
        help="early stopping after N epochs without validation accuracy improving",
    )
    parser.add_argument(
        "--image-size", default=None, type=int, help="resolution of image"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )

    parser.add_argument("--end-lr", default=0, type=float)

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )
    parser.add_argument(
        "--optimizer", default="sgd", type=str, choices=("sgd", "rmsprop")
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--rmsprop-alpha",
        default=0.9,
        type=float,
        help="value of alpha parameter in rmsprop optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--rmsprop-eps",
        default=1e-3,
        type=float,
        help="value of eps parameter in rmsprop optimizer (default: 1e-3)",
    )

    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve amp convergence.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        default="0",
        type=int,
        help=(
            "Gather N last checkpoints throughout the training,"
            " without this flag only best and last checkpoints will be stored. "
            "Use -1 for all checkpoints"
        ),
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )
    parser.add_argument(
        "--jit",
        type=str,
        default="no",
        choices=["no", "script"],
        help="no -> do not use torch.jit; script -> use torch.jit.script",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)

    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
    parser.add_argument("--use-ema", default=None, type=float, help="use EMA")
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        choices=[None, "autoaugment"],
        help="augmentation method",
    )

    parser.add_argument(
        "--gpu-affinity",
        type=str,
        default="none",
        required=False,
        choices=[am.name for am in AffinityMode],
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        required=False,
    )


def prepare_env(args):
    # seed_everything(args.seed)
    
    save_dir = os.path.join(args.save_dir, args.case_name)
    
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_dir, f"experiment.log"),
                        format='%(asctime)s %(message)s',
                        filemode='wt',
                        level=logging.INFO)
    
    
def prepare_dataset(args):
    # prerare dataset
    print("prepare dataset")
    train_csv = os.path.join(args.data_dir, "test.csv")
    train_images = os.path.join(args.data_dir, "test_images")
    training_data = get_training_data(train_csv, train_images)

    total_num = len(training_data)
    train_num = int(total_num * args.train_ratio) + 1
    val_num = int(total_num * args.valid_ratio)

    train_cases = training_data[:train_num]
    val_cases = training_data[train_num: train_num+val_num]
    test_cases  = training_data[train_num+val_num:]

    print("number of training case", len(train_cases))
    print("number of validation case", len(val_cases))
    print("number of testing case", len(test_cases))

    return (train_cases, val_cases, test_cases)

def prepare_loader(args, train_cases, val_cases):
    print("prepare train loader")
    train_loader = DataLoader(train_cases,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CaseCollator())

def prepare_train_loader(args):   
    train_loader, train_loader_len = get_pytorch_train_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        args.mixup > 0.0,
        interpolation=args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)
    return (train_loader, train_loader_len)

def prepare_val_loader(args):
     val_loader, val_loader_len = get_val_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )
     return (val_loader, val_loader_len)
    
    
def prepare_training(args, model_arch):
    prepare_env(args)
    start_epoch = 0
    best_prec1 = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            if "state_dict_ema" in checkpoint:
                model_state_ema = checkpoint["state_dict_ema"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if start_epoch >= args.epochs:
                print(
                    f"Launched training for {args.epochs}, checkpoint already run {start_epoch}"
                )
                exit(1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            model_state_ema = None
            optimizer_state = None
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    train_cases, val_cases, test_cases = prepare_dataset(args)

    train_loader, val_loader = prepare_loader(args, train_cases)


    # model = model_arch(
    #     **{
    #         k: v
    #         if k != "pretrained"
    #         else v and (not args.distributed or dist.get_rank() == 0)
    #         for k, v in model_args.__dict__.items()
    #     }
    # )
    loss = nn.MSELoss(reduction="mean")
    
    train_loader, train_loader_len = prepare_train_loader(args)
    val_loader, val_loader_len = prepare_val_loader(args)
    
    model = model_arch()
    
    image_size = (
        args.image_size
        if args.image_size is not None
        else model.arch.default_image_size
    )

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100,
        enabled=args.amp,
    )
    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )
    
    executor = Executor(
        model,
        loss(),
        cuda=True,
        memory_format=memory_format,
        amp=args.amp,
        scaler=scaler,
        divide_loss=batch_size_multiplier,
        ts_script=args.jit == "script",
    )
    
    optimizer = get_optimizer(
        list(executor.model.named_parameters()),
        args.lr,
        args=args,
        state=optimizer_state,
    )
    
    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup)
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(
            args.lr, args.warmup, args.epochs, end_lr=args.end_lr
        )
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs)

    if args.distributed:
        executor.distributed(args.gpu)

    if model_state is not None:
        executor.model.load_state_dict(model_state)

    trainer = Trainer(
        executor,
        optimizer,
        grad_acc_steps=batch_size_multiplier,
        ema=args.use_ema,
    )

    if (args.use_ema is not None) and (model_state_ema is not None):
        trainer.ema_executor.model.load_state_dict(model_state_ema)

    return (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
        best_prec1,
    )


    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser) 
    args = parser.parse_args()
    
    model_arch = available_models()[args.archs]
    
    (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
        best_prec1,
    ) = prepare_training(args, model_arch)

    train_loop(
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs)
        if args.run_epochs != -1
        else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
        keep_last_n_checkpoints=args.gather_checkpoints,
        topk=args.topk,
    )

    
    # prepare_training(args, model_arch)


    # model.to(args.device)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)
    # criterion = nn.MSELoss(reduction='mean')

    # for e in range(1, args.epochs + 1):``
    #     print("epoch", e)
    #     train_epoch(e, train_loader, model, optimizer, criterion)
    #     if e % args.eval_every == 0 or e == args.epochs:
    #         print("on valid cases")
    #         metric_data = evaluate(valid_cases, model, criterion, e)
    #         pd.DataFrame(metric_data).to_csv(
    #             os.path.join(save_dir, f"valid_metric_{e}.csv"))

    #         print("on test cases")
    #         metric_data = evaluate(test_cases, model, criterion, e)
    #         pd.DataFrame(metric_data).to_csv(
    #             os.path.join(save_dir, f"test_metric_{e}.csv"))

    #     if e % args.save_every == 0:
    #         torch.save(model.state_dict(), os.path.join(save_dir, f"model_{e}.pth"))