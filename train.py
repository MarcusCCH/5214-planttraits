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

from utils.data import get_training_data, CaseCollator


from image_classification.models import (
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
    parser.add_argument(
        "--model",
        "-m",
        default="efficientnet_b4",
        choices=model_names,
        help="model architecture: "
        + " | ".join(model_names)
        + " (default: resnet50)",
    )


    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='data/')

    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)

    save_dir = os.path.join(args.save_dir, args.case_name)
    
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_dir, f"experiment.log"),
                        format='%(asctime)s %(message)s',
                        filemode='wt',
                        level=logging.INFO)

    # prerare dataset
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_images = os.path.join(args.data_dir, "train_images")
    training_data = get_training_data(train_csv, train_images)

    total_num = len(training_data)
    train_num = int(total_num * args.train_ratio) + 1
    valid_num = int(total_num * args.valid_ratio)

    train_cases = training_data[:train_num]
    valid_cases = training_data[train_num: train_num+valid_num]
    test_cases  = training_data[train_num+valid_num:]

    print("number of training case", len(train_cases))
    print("number of validation case", len(valid_cases))
    print("number of testing case", len(test_cases))

    train_loader = DataLoader(train_cases,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CaseCollator())

    print("prepare train loader")

    model = available_models()[args.model]

    model.to(args.device)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)
    # criterion = nn.MSELoss(reduction='mean')

    # for e in range(1, args.epochs + 1):
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