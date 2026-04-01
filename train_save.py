"""
Training script for the SAVE syscall log dataset.
Uses back_3.10-slim-bullseye.txt (benign only) to train the GMAE model.

Usage:
    python train_save.py
    python train_save.py --window_size 50 --stride 10 --batch_size 32 --max_epoch 10
"""

import os
import random
import argparse
import warnings
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

from utils.loaddata import load_save_dataset
from model.autoencoder import build_model
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer

warnings.filterwarnings('ignore')

LOG_PATH = './data/SAVE/back_3.10-slim-bullseye.txt'
CACHE_PATH = './data/SAVE/cache_back_3.10-slim-bullseye.pkl'
CHECKPOINT_PATH = './checkpoints/checkpoint-save.pt'


def build_args():
    parser = argparse.ArgumentParser(description="MAGIC - SAVE syscall dataset")
    # Data
    parser.add_argument('--log_path', type=str, default=LOG_PATH)
    parser.add_argument('--cache_path', type=str, default=CACHE_PATH)
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--window_size', type=int, default=50,
                        help='Sliding window size for syscall sequences')
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride for sliding window')
    parser.add_argument('--syscall_dim', type=int, default=356,
                        help='One-hot feature dimension = max syscall number + 1 (kernel 6.8: 356)')
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=-1,
                        help='GPU id, -1 for CPU')
    # Model
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mask_rate', type=float, default=0.5)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--alpha_l', type=float, default=3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--loss_fn', type=str, default='sce')
    parser.add_argument('--pooling', type=str, default='mean')
    return parser.parse_args()


def main():
    args = build_args()
    device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    set_random_seed(0)

    # ── Load & preprocess dataset ──────────────────────────────────────────
    dataset = load_save_dataset(
        log_path=args.log_path,
        window_size=args.window_size,
        stride=args.stride,
        cache_path=args.cache_path,
        syscall_dim=args.syscall_dim,
    )

    graphs = dataset['dataset']
    train_index = dataset['train_index']
    args.n_dim = dataset['n_feat']
    args.e_dim = dataset['e_feat']

    # ── Dataloader ─────────────────────────────────────────────────────────
    random.shuffle(train_index)
    train_idx = torch.arange(len(train_index))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(
        train_index, batch_size=args.batch_size, sampler=train_sampler
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(args)
    model = model.to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    # ── Train ──────────────────────────────────────────────────────────────
    model = batch_level_train(
        model, graphs, train_loader,
        optimizer, args.max_epoch, device,
        args.n_dim, args.e_dim,
    )

    # ── Save checkpoint ────────────────────────────────────────────────────
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f'Checkpoint saved to {args.checkpoint_path}')


if __name__ == '__main__':
    main()
