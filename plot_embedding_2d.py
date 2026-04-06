"""
2D embedding visualization using t-SNE for model drift motivation figure.

eval_save.py 대신 이 스크립트가 직접 임베딩을 추출하고 t-SNE로 시각화한다.

Usage:
    python plot_embedding_2d.py

설정은 아래 CONFIG 블록만 수정하면 됨.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.loaddata import load_save_dataset, transform_graph
from utils.save_parser import preprocess_save_dataset
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed

warnings.filterwarnings('ignore')

# ── 설정 (여기만 수정) ──────────────────────────────────────────────────────

CHECKPOINT   = './checkpoints/online_boutique/v0.3.6/adservice.pt'
OUT_DIR      = './figs'
DEVICE       = 'cuda:0'
SYSCALL_DIM  = 449

# 모델 구조 (train_save.py와 동일하게)
NUM_HIDDEN     = 64
NUM_LAYERS     = 3
NEGATIVE_SLOPE = 0.2
MASK_RATE      = 0.5
ALPHA_L        = 3
POOLING        = 'mean'
WINDOW_SIZE    = 50
STRIDE         = 10

# 시각화할 데이터 목록
# label      : 범례에 표시할 이름
# path       : 로그 파일 경로
# cache      : 파싱 캐시 경로
# kind       : 'benign' or 'attack'
# max_samples: None이면 전체 사용, 정수면 해당 수만큼 랜덤 샘플링 (t-SNE 속도용)
SOURCES = [
    dict(label='v0.3.6 (train)', path='./data/online_boutique/v0.3.6/adservice.txt',
         cache='./data/online_boutique/v0.3.6/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.4.0',         path='./data/online_boutique/v0.4.0/adservice.txt',
         cache='./data/online_boutique/v0.4.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.5.0',         path='./data/online_boutique/v0.5.0/adservice.txt',
         cache='./data/online_boutique/v0.5.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.6.0',         path='./data/online_boutique/v0.6.0/adservice.txt',
         cache='./data/online_boutique/v0.6.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.7.0',         path='./data/online_boutique/v0.7.0/adservice.txt',
         cache='./data/online_boutique/v0.7.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.8.0',         path='./data/online_boutique/v0.8.0/adservice.txt',
         cache='./data/online_boutique/v0.8.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.9.0',         path='./data/online_boutique/v0.9.0/adservice.txt',
         cache='./data/online_boutique/v0.9.0/adservice.pkl',  kind='benign', max_samples=2000),
    dict(label='v0.10.5',        path='./data/online_boutique/v0.10.5/adservice.txt',
         cache='./data/online_boutique/v0.10.5/adservice.pkl', kind='benign', max_samples=2000),
    dict(label='Attack',         path='./data/Attack/',
         cache=None,                                            kind='attack', max_samples=2000),
]

# 색상 + 마커: 흑백 인쇄에서도 구분되도록
# benign 버전은 파란 계열 + 마커 변화, attack은 빨간 X
_BENIGN_STYLES = [
    dict(color='#08306B', marker='o'),
    dict(color='#2171B5', marker='s'),
    dict(color='#4292C6', marker='^'),
    dict(color='#6BAED6', marker='D'),
    dict(color='#9ECAE1', marker='v'),
    dict(color='#C6DBEF', marker='P'),
    dict(color='#DEEBF7', marker='*'),
    dict(color='#888888', marker='X'),
]
_ATTACK_STYLE = dict(color='#CB181D', marker='x')

# t-SNE 파라미터
TSNE_PERPLEXITY  = 50
TSNE_N_ITER      = 2000
TSNE_RANDOM_STATE = 42

# ────────────────────────────────────────────────────────────────────────────


class _Args:
    """build_model이 요구하는 args 객체."""
    def __init__(self, n_dim, e_dim):
        self.num_hidden     = NUM_HIDDEN
        self.num_layers     = NUM_LAYERS
        self.negative_slope = NEGATIVE_SLOPE
        self.mask_rate      = MASK_RATE
        self.alpha_l        = ALPHA_L
        self.n_dim          = n_dim
        self.e_dim          = e_dim


@torch.no_grad()
def _embed_batch(model, batch_gs, device, pooler):
    import dgl
    bg = dgl.batch(batch_gs).to(device)
    out = model.embed(bg)
    gs = dgl.unbatch(bg)
    offset = 0
    result = []
    for g in gs:
        n = g.num_nodes()
        result.append(pooler(g, out[offset:offset + n]).cpu().numpy())
        offset += n
    return result


@torch.no_grad()
def extract_embeddings(model, graphs_or_raw, n_dim, e_dim, device, pooler,
                       is_raw=False, batch_size=256, max_samples=None):
    """그래프 리스트에서 임베딩 벡터 추출."""
    model.eval()

    if is_raw:
        items = graphs_or_raw
    else:
        items = graphs_or_raw

    if max_samples and len(items) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(items), max_samples, replace=False)
        items = [items[i] for i in sorted(idx)]

    embeddings = []
    batch_gs = []

    for item in tqdm(items, desc='  Embedding', leave=False):
        if is_raw:
            g = transform_graph(item, n_dim, e_dim)
        else:
            g = transform_graph(item[0], n_dim, e_dim)
        batch_gs.append(g)
        if len(batch_gs) == batch_size:
            embeddings.extend(_embed_batch(model, batch_gs, device, pooler))
            batch_gs = []
    if batch_gs:
        embeddings.extend(_embed_batch(model, batch_gs, device, pooler))

    return np.concatenate(embeddings, axis=0)


def load_source(src, n_dim, e_dim, device, model, pooler):
    """SOURCES 항목 하나를 로드해 임베딩 반환."""
    kind = src['kind']

    if kind == 'attack':
        # 폴더 내 .txt 전부
        folder = src['path']
        import dgl
        all_graphs = []
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(folder, fname)
            cache = fpath.replace('.txt', '_cache.pkl')
            graphs, _ = preprocess_save_dataset(fpath, WINDOW_SIZE, STRIDE, cache, SYSCALL_DIM)
            all_graphs.extend(graphs)
        emb = extract_embeddings(model, all_graphs, n_dim, e_dim, device, pooler,
                                 is_raw=True, max_samples=src['max_samples'])
    else:
        data = load_save_dataset(src['path'], WINDOW_SIZE, STRIDE,
                                 src['cache'], SYSCALL_DIM)
        graphs = data['dataset']
        items = [graphs[i] for i in range(len(graphs))]
        emb = extract_embeddings(model, items, n_dim, e_dim, device, pooler,
                                 is_raw=False, max_samples=src['max_samples'])
    return emb


def run_tsne(embeddings_list):
    """모든 임베딩을 합쳐 t-SNE 실행 후 분리 반환."""
    sizes = [len(e) for e in embeddings_list]
    X = np.concatenate(embeddings_list, axis=0)
    print(f'\nRunning t-SNE on {len(X)} samples (perplexity={TSNE_PERPLEXITY}, n_iter={TSNE_N_ITER}) ...')
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER,
                random_state=TSNE_RANDOM_STATE, verbose=1)
    X2d = tsne.fit_transform(X)

    result = []
    offset = 0
    for s in sizes:
        result.append(X2d[offset:offset + s])
        offset += s
    return result


def assign_styles(sources):
    styles = []
    benign_idx = 0
    for src in sources:
        if src['kind'] == 'attack':
            styles.append(_ATTACK_STYLE)
        else:
            styles.append(_BENIGN_STYLES[benign_idx % len(_BENIGN_STYLES)])
            benign_idx += 1
    return styles


def plot_2d(coords_list, sources, styles, save_path, alpha_benign=0.4, alpha_attack=0.6,
            point_size=8):
    fig, ax = plt.subplots(figsize=(8, 6))

    handles = []
    for coords, src, style in zip(coords_list, sources, styles):
        alpha = alpha_attack if src['kind'] == 'attack' else alpha_benign
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=style['color'], marker=style['marker'],
                        s=point_size, alpha=alpha, linewidths=0,
                        label=src['label'])
        handles.append(mpatches.Patch(color=style['color'], label=src['label']))

    ax.legend(handles=handles, fontsize=8, loc='best',
              framealpha=0.8, markerscale=1.5)
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.set_title('Embedding Distribution (t-SNE)', fontsize=13)
    ax.set_facecolor('white')
    ax.grid(linestyle=':', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def main():
    set_random_seed(42)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    # 모델 로드 (n_dim/e_dim은 첫 번째 benign 소스에서 결정)
    first_benign = next(s for s in SOURCES if s['kind'] == 'benign')
    data = load_save_dataset(first_benign['path'], WINDOW_SIZE, STRIDE,
                             first_benign['cache'], SYSCALL_DIM)
    n_dim, e_dim = data['n_feat'], data['e_feat']

    args = _Args(n_dim, e_dim)
    model = build_model(args)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)
    pooler = Pooling(POOLING)
    print(f'Model loaded: {CHECKPOINT}')

    # 각 소스 임베딩 추출
    embeddings_list = []
    for src in SOURCES:
        print(f"\n[{src['label']}] {src['path']}")
        emb = load_source(src, n_dim, e_dim, device, model, pooler)
        embeddings_list.append(emb)
        print(f"  → {emb.shape}")

    # t-SNE
    coords_list = run_tsne(embeddings_list)

    # 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    npy_path = os.path.join(OUT_DIR, 'tsne_coords.npy')
    labels_path = os.path.join(OUT_DIR, 'tsne_labels.npy')
    np.save(npy_path, np.concatenate(coords_list, axis=0))
    sizes = [len(c) for c in coords_list]
    np.save(labels_path, np.array(sizes))
    print(f'Coords saved: {npy_path}')

    # 시각화
    styles = assign_styles(SOURCES)
    plot_2d(coords_list, SOURCES, styles,
            os.path.join(OUT_DIR, 'fig_embedding_tsne.pdf'))


if __name__ == '__main__':
    main()
