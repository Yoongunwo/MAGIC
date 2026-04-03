"""
Evaluation script for the SAVE syscall log dataset.

두 가지 모드:
1. 공격 로그 없음 (benign-only):
   - back_3.10-slim-bullseye.txt를 train/test로 분할
   - KNN 이상 점수 분포만 출력 (AUC 계산 불가)

2. 공격 로그 지정 (--attack_paths):
   - benign train으로 KNN 피팅 → benign test + attack test로 AUC/F1/Precision/Recall 산출
   - 파일 개별 지정 또는 폴더 지정 모두 가능 (폴더 지정 시 .txt 파일 전부 사용)

Usage:
    # benign-only mode
    python eval_save.py

    # 공격 로그 파일 직접 지정
    python eval_save.py --attack_paths data/SAVE/attack1.txt data/SAVE/attack2.txt

    # 공격 로그가 들어있는 폴더 지정
    python eval_save.py --attack_paths data/SAVE/attacks/

    # 파일 + 폴더 혼합도 가능
    python eval_save.py --attack_paths data/SAVE/attacks/ data/SAVE/extra_attack.txt

    # 파라미터 조정
    python eval_save.py --train_ratio 0.8 --repeat 10 --attack_paths data/SAVE/attacks/
"""

import os
import random
import argparse
import warnings
import pickle as pkl

import torch
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from utils.loaddata import load_save_dataset, transform_graph
from utils.save_parser import preprocess_save_dataset
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed

warnings.filterwarnings('ignore')

BENIGN_LOG = './data/SAVE/back_3.10-slim-bullseye.txt'
BENIGN_CACHE = './data/SAVE/cache_back_3.10-slim-bullseye.pkl'
CHECKPOINT = './checkpoints/checkpoint-save.pt'


def build_args():
    parser = argparse.ArgumentParser(description='MAGIC - SAVE Evaluation')
    parser.add_argument('--benign_path', type=str, default=BENIGN_LOG)
    parser.add_argument('--benign_cache', type=str, default=BENIGN_CACHE)
    parser.add_argument('--ref_path', type=str, default=None,
                        help='KNN fitting용 레퍼런스 benign 로그 (지정 시 benign_path는 전체가 test가 됨). '
                             'Model drift 측정에 사용. 미지정 시 benign_path를 train_ratio로 분할.')
    parser.add_argument('--ref_cache', type=str, default=None,
                        help='ref_path의 파싱 캐시 경로')
    parser.add_argument('--attack_paths', type=str, nargs='*', default=[],
                        help='Paths to attack log files or directories containing .txt logs (label=1). Optional.')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of benign graphs used as train split (ref_path 미지정 시에만 사용)')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=10,
                        help='Repetitions for AUC averaging (only used with attack logs)')
    parser.add_argument('--syscall_dim', type=int, default=449,
                        help='One-hot feature dimension = max syscall number + 1 (kernel 6.8: 356)')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--mask_rate', type=float, default=0.5)
    parser.add_argument('--alpha_l', type=float, default=3)
    return parser.parse_args()


@torch.no_grad()
def _embed_batch(model, batch_gs, device, pooler):
    """Embed a list of already-transformed DGL graphs as one batch."""
    bg = dgl.batch(batch_gs).to(device)
    out = model.embed(bg)
    # unbatch: 각 그래프별로 pooling
    gs = dgl.unbatch(bg)
    offset = 0
    result = []
    for g in gs:
        n = g.num_nodes()
        result.append(pooler(g, out[offset:offset + n]).cpu().numpy())
        offset += n
    return result


@torch.no_grad()
def embed_graphs(model, graphs, indices, n_dim, e_dim, device, pooler, batch_size=256):
    """Return (N, D) embedding matrix for the given graph indices."""
    model.eval()
    embeddings = []
    batch_gs = []
    for idx in tqdm(indices, desc='Embedding'):
        batch_gs.append(transform_graph(graphs[idx][0], n_dim, e_dim))
        if len(batch_gs) == batch_size:
            embeddings.extend(_embed_batch(model, batch_gs, device, pooler))
            batch_gs = []
    if batch_gs:
        embeddings.extend(_embed_batch(model, batch_gs, device, pooler))
    return np.concatenate(embeddings, axis=0)


@torch.no_grad()
def embed_raw_graphs(model, raw_graphs, n_dim, e_dim, device, pooler, batch_size=256):
    """Embed a plain list of DGL graphs (no label wrapper)."""
    model.eval()
    embeddings = []
    batch_gs = []
    for g in tqdm(raw_graphs, desc='Embedding attack graphs'):
        batch_gs.append(transform_graph(g, n_dim, e_dim))
        if len(batch_gs) == batch_size:
            embeddings.extend(_embed_batch(model, batch_gs, device, pooler))
            batch_gs = []
    if batch_gs:
        embeddings.extend(_embed_batch(model, batch_gs, device, pooler))
    return np.concatenate(embeddings, axis=0)


def knn_anomaly_score(x_train, x_query, n_neighbors):
    """Fit KNN on x_train, return per-sample anomaly scores for x_query."""
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-6
    x_train_norm = (x_train - x_mean) / x_std
    x_query_norm = (x_query - x_mean) / x_std

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(x_train_norm)

    # Baseline: average NN distance within training set (leave-one-out approximation)
    ref_dists, _ = nbrs.kneighbors(x_train_norm)
    mean_dist = ref_dists.mean() * n_neighbors / max(n_neighbors - 1, 1)

    query_dists, _ = nbrs.kneighbors(x_query_norm)
    scores = query_dists.mean(axis=1) / (mean_dist + 1e-9)
    return scores


def evaluate_with_labels(x_train, x_benign_test, x_attack, n_neighbors, repeat):
    x_test = np.concatenate([x_benign_test, x_attack], axis=0)
    y_test = np.concatenate([
        np.zeros(len(x_benign_test)),
        np.ones(len(x_attack))
    ])

    benign_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]

    auc_list, f1_list, prec_list, rec_list = [], [], [], []

    for s in range(repeat):
        set_random_seed(s)
        np.random.shuffle(benign_idx)

        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0) + 1e-6
        x_tr = (x_train - x_mean) / x_std
        x_te = (x_test - x_mean) / x_std

        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_tr)

        ref_dists, _ = nbrs.kneighbors(x_tr)
        mean_dist = ref_dists.mean() * n_neighbors / max(n_neighbors - 1, 1)
        query_dists, _ = nbrs.kneighbors(x_te)
        scores = query_dists.mean(axis=1) / (mean_dist + 1e-9)

        auc = roc_auc_score(y_test, scores)
        prec, rec, thresholds = precision_recall_curve(y_test, scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = np.argmax(f1)

        auc_list.append(auc)
        f1_list.append(f1[best])
        prec_list.append(prec[best])
        rec_list.append(rec[best])

    print(f'AUC       : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')
    print(f'F1        : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}')
    print(f'Precision : {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}')
    print(f'Recall    : {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}')
    return np.mean(auc_list), np.std(auc_list)


def evaluate_benign_only(scores):
    print('--- Anomaly Score Distribution (benign test) ---')
    print(f'  mean  : {scores.mean():.4f}')
    print(f'  std   : {scores.std():.4f}')
    print(f'  min   : {scores.min():.4f}')
    print(f'  median: {np.median(scores):.4f}')
    print(f'  95th  : {np.percentile(scores, 95):.4f}')
    print(f'  max   : {scores.max():.4f}')


def main():
    args = build_args()
    device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    set_random_seed(0)

    # ── Load benign (test) dataset ─────────────────────────────────────────
    benign_data = load_save_dataset(
        log_path=args.benign_path,
        window_size=args.window_size,
        stride=args.stride,
        cache_path=args.benign_cache,
        syscall_dim=args.syscall_dim,
    )
    graphs = benign_data['dataset']
    n_dim = benign_data['n_feat']
    e_dim = benign_data['e_feat']

    # ── Load model ─────────────────────────────────────────────────────────
    args.n_dim = n_dim
    args.e_dim = e_dim
    model = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    pooler = Pooling(args.pooling)

    # ── KNN fitting 데이터 결정 ────────────────────────────────────────────
    if args.ref_path:
        # Model drift 모드: ref_path 전체로 KNN fit, benign_path 전체를 test로
        ref_data = load_save_dataset(
            log_path=args.ref_path,
            window_size=args.window_size,
            stride=args.stride,
            cache_path=args.ref_cache,
            syscall_dim=args.syscall_dim,
        )
        ref_graphs = ref_data['dataset']
        ref_idx = list(range(len(ref_graphs)))
        test_benign_idx = list(range(len(graphs)))
        print(f'[Drift mode] KNN ref: {len(ref_idx)} ({args.ref_path})')
        print(f'[Drift mode] Benign test: {len(test_benign_idx)} ({args.benign_path})')
        x_train = embed_graphs(model, ref_graphs, ref_idx, n_dim, e_dim, device, pooler)
        x_benign_test = embed_graphs(model, graphs, test_benign_idx, n_dim, e_dim, device, pooler)
    else:
        # 기본 모드: benign_path를 train_ratio로 분할
        all_idx = list(range(len(graphs)))
        random.shuffle(all_idx)
        split = int(len(all_idx) * args.train_ratio)
        train_idx = all_idx[:split]
        test_benign_idx = all_idx[split:]
        print(f'Benign train: {len(train_idx)}  |  Benign test: {len(test_benign_idx)}')
        x_train = embed_graphs(model, graphs, train_idx, n_dim, e_dim, device, pooler)
        x_benign_test = embed_graphs(model, graphs, test_benign_idx, n_dim, e_dim, device, pooler)

    # ── Load & embed attack graphs (optional) ──────────────────────────────
    if args.attack_paths:
        # Expand folders to individual .txt files
        resolved_attack_files = []
        for path in args.attack_paths:
            if os.path.isdir(path):
                txt_files = sorted(
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith('.txt')
                )
                if not txt_files:
                    print(f'  Warning: no .txt files found in directory {path}')
                else:
                    print(f'  Found {len(txt_files)} attack log(s) in {path}:')
                    for f in txt_files:
                        print(f'    {f}')
                resolved_attack_files.extend(txt_files)
            else:
                resolved_attack_files.append(path)

        attack_raw_graphs = []
        for path in resolved_attack_files:
            cache = path.replace('.txt', '_cache.pkl')
            ag, max_sc = preprocess_save_dataset(path, args.window_size, args.stride, cache, args.syscall_dim)
            # Expand feature dim if attack logs contain unseen syscall numbers
            if max_sc + 1 > n_dim:
                print(f'  Warning: syscall {max_sc} exceeds SYSCALL_DIM {n_dim}. '
                      f'Consider raising SYSCALL_DIM in save_parser.py.')
            attack_raw_graphs.extend(ag)
        print(f'Attack graphs: {len(attack_raw_graphs)}')
        x_attack = embed_raw_graphs(model, attack_raw_graphs, n_dim, e_dim, device, pooler)

        print('\n=== Evaluation (with attack labels) ===')
        evaluate_with_labels(x_train, x_benign_test, x_attack, args.n_neighbors, args.repeat)
    else:
        print('\n=== Evaluation (benign-only, no attack labels) ===')
        scores = knn_anomaly_score(x_train, x_benign_test, args.n_neighbors)
        evaluate_benign_only(scores)


if __name__ == '__main__':
    main()
