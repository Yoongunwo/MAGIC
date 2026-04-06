import os
import pickle as pkl
import torch
import dgl
from collections import defaultdict
from tqdm import tqdm

# Linux x86-64 syscall 번호 상한 (커널 6.8 기준 최대 355번 → dim=356).
# 학습/평가 데이터에 관계없이 동일한 입력 차원을 보장한다.
SYSCALL_DIM = 356


def parse_log_file(filepath):
    """
    Parse log file with format: timestamp PID=xxx syscall=xxx
    Returns list of (timestamp, pid, syscall) tuples, already in time order.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                timestamp = float(parts[0])
                pid = int(parts[1].split('=')[1])
                syscall = int(parts[2].split('=')[1])
                records.append((timestamp, pid, syscall))
            except (ValueError, IndexError):
                continue
    return records


def group_by_pid(records):
    """
    Group syscall records by PID.
    Records are assumed to be in time order (as parsed from the log).
    Returns dict: pid -> [syscall, ...]
    """
    pid_sequences = defaultdict(list)
    for _, pid, syscall in records:
        pid_sequences[pid].append(syscall)
    return pid_sequences


def build_syscall_graph(window):
    """
    Build a DGL graph from a syscall window.
    - Nodes: unique syscall numbers in the window (node type = syscall number)
    - Edges: directed transition edges syscall[i] -> syscall[i+1]
    - Edge type: 0 (single transition type)
    Returns None if no edges can be formed.
    """
    # Deduplicate while preserving order for stable node indexing
    seen = {}
    for s in window:
        if s not in seen:
            seen[s] = len(seen)
    syscall_to_idx = seen

    src_nodes = []
    dst_nodes = []
    for i in range(len(window) - 1):
        src_nodes.append(syscall_to_idx[window[i]])
        dst_nodes.append(syscall_to_idx[window[i + 1]])

    if not src_nodes:
        return None

    node_syscalls = list(seen.keys())  # syscall number per node index

    g = dgl.graph((src_nodes, dst_nodes))
    g.ndata['type'] = torch.tensor(node_syscalls, dtype=torch.long)
    g.edata['type'] = torch.zeros(len(src_nodes), dtype=torch.long)
    return g


def build_graphs_sliding_window(pid_sequences, window_size=50, stride=10, dedup=True):
    """
    For each PID's syscall sequence, apply a sliding window and build
    a DGL graph per window.
    dedup=True이면 동일한 syscall 시퀀스(tuple)는 한 번만 그래프로 만든다.
    Returns list of DGL graphs.
    """
    seen_windows = set() if dedup else None
    graphs = []
    for pid, syscalls in tqdm(pid_sequences.items(), desc="Building graphs per PID"):
        if len(syscalls) < window_size:
            continue
        for i in range(0, len(syscalls) - window_size + 1, stride):
            window = tuple(syscalls[i:i + window_size])
            if dedup:
                if window in seen_windows:
                    continue
                seen_windows.add(window)
            g = build_syscall_graph(window)
            if g is not None:
                graphs.append(g)
    return graphs


def preprocess_save_dataset(filepath, window_size=50, stride=10, cache_path=None, syscall_dim=SYSCALL_DIM):
    """
    Full preprocessing pipeline for a SAVE-format log file.
    If cache_path is given and exists, loads cached graphs directly.
    Returns list of DGL graphs and the max syscall number seen.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pkl.load(f)
        cached_max = data['max_syscall']
        if cached_max >= syscall_dim:
            raise ValueError(
                f"Cache {cache_path} has max_syscall={cached_max} but syscall_dim={syscall_dim}. "
                f"Delete the cache and re-run with --syscall_dim {cached_max + 1}."
            )
        return data['graphs'], cached_max

    print(f"Parsing {filepath} ...")
    records = parse_log_file(filepath)
    print(f"  Total records   : {len(records)}")

    pid_sequences = group_by_pid(records)
    print(f"  Unique PIDs     : {len(pid_sequences)}")

    graphs = build_graphs_sliding_window(pid_sequences, window_size, stride)
    print(f"  Total graphs    : {len(graphs)}")

    # 실제 데이터의 max syscall 확인 및 검증
    actual_max = 0
    for g in graphs:
        if g.num_nodes() > 0:
            actual_max = max(actual_max, g.ndata['type'].max().item())
    print(f"  Max syscall     : {actual_max}  (syscall_dim={syscall_dim})")
    if actual_max >= syscall_dim:
        raise ValueError(
            f"Data contains syscall {actual_max} but syscall_dim={syscall_dim}. "
            f"Re-run with --syscall_dim {actual_max + 1} (and delete the cache first)."
        )

    max_syscall = actual_max  # 실제 최댓값 저장

    data = {'graphs': graphs, 'max_syscall': max_syscall}

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pkl.dump(data, f)
        print(f"  Cached to       : {cache_path}")

    return graphs, max_syscall
