"""
Syscall sequence overlap analysis across versions / files.

슬라이딩 윈도우(size=50)로 추출한 syscall 시퀀스를 tuple로 표현하여
두 파일(또는 여러 파일) 간 동일 시퀀스 / 고유 시퀀스 분포를 분석한다.

Usage:
    # 두 버전 비교
    python analyze_sequence_overlap.py \
        --files ./data/online_boutique/v0.3.6/adservice.txt \
                ./data/online_boutique/v0.10.5/adservice.txt \
        --labels v0.3.6 v0.10.5

    # 여러 파일 한 번에
    python analyze_sequence_overlap.py \
        --files ./data/SAVE/back_3.10-slim-bullseye.txt \
                ./data/SAVE/back_3.10-slim-bookworm.txt \
                ./data/SAVE/back_3.10-alpine.txt \
        --labels bullseye bookworm alpine
"""

import argparse
from collections import Counter, defaultdict
from utils.save_parser import parse_log_file, group_by_pid


WINDOW_SIZE = 50
STRIDE = 1  # stride=1이어야 모든 가능한 시퀀스를 수집


def extract_sequences(filepath):
    """
    파일에서 PID별 슬라이딩 윈도우 시퀀스를 tuple 리스트로 반환.
    중복 포함 전체 시퀀스와 Counter(빈도)를 함께 반환.
    """
    records = parse_log_file(filepath)
    pid_sequences = group_by_pid(records)

    all_seqs = []
    for pid, syscalls in pid_sequences.items():
        if len(syscalls) < WINDOW_SIZE:
            continue
        for i in range(0, len(syscalls) - WINDOW_SIZE + 1, STRIDE):
            all_seqs.append(tuple(syscalls[i:i + WINDOW_SIZE]))

    counter = Counter(all_seqs)
    return all_seqs, counter


def analyze_pair(label_a, counter_a, label_b, counter_b):
    """두 파일 간 시퀀스 overlap 분석."""
    set_a = set(counter_a.keys())
    set_b = set(counter_b.keys())

    only_a = set_a - set_b
    only_b = set_b - set_a
    common = set_a & set_b

    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())

    # common 시퀀스가 각 파일에서 차지하는 비중 (occurrence 기준)
    common_occ_a = sum(counter_a[s] for s in common)
    common_occ_b = sum(counter_b[s] for s in common)

    print(f'\n{"="*60}')
    print(f'  {label_a}  vs  {label_b}')
    print(f'{"="*60}')
    print(f'{"":30s} {label_a:>12s} {label_b:>12s}')
    print(f'{"-"*60}')
    print(f'{"총 시퀀스 수 (중복포함)":30s} {total_a:>12,} {total_b:>12,}')
    print(f'{"고유 시퀀스 종류":30s} {len(set_a):>12,} {len(set_b):>12,}')
    print(f'{"공통 시퀀스 종류":30s} {len(common):>12,}')
    print(f'{"A에만 있는 시퀀스 종류":30s} {len(only_a):>12,}')
    print(f'{"B에만 있는 시퀀스 종류":30s} {len(only_b):>12,}')
    print(f'{"-"*60}')

    jaccard = len(common) / len(set_a | set_b) if set_a | set_b else 0.0
    print(f'{"Jaccard 유사도 (종류 기준)":30s} {jaccard:>12.4f}')

    overlap_ratio_a = common_occ_a / total_a if total_a else 0.0
    overlap_ratio_b = common_occ_b / total_b if total_b else 0.0
    print(f'{"공통 시퀀스 비중 (A 발생 기준)":30s} {overlap_ratio_a:>12.4f}')
    print(f'{"공통 시퀀스 비중 (B 발생 기준)":30s} {overlap_ratio_b:>12.4f}')

    # A에만 있는 시퀀스 상위 10개
    if only_a:
        print(f'\n  [{label_a}에만 있는 상위 10개 시퀀스 (빈도순)]')
        for seq, cnt in sorted(((s, counter_a[s]) for s in only_a),
                               key=lambda x: -x[1])[:10]:
            print(f'    count={cnt:6d}  {list(seq[:10])}...')

    if only_b:
        print(f'\n  [{label_b}에만 있는 상위 10개 시퀀스 (빈도순)]')
        for seq, cnt in sorted(((s, counter_b[s]) for s in only_b),
                               key=lambda x: -x[1])[:10]:
            print(f'    count={cnt:6d}  {list(seq[:10])}...')


def analyze_all(labels, counters):
    """여러 파일 간 pairwise + 전체 요약."""
    n = len(labels)

    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            analyze_pair(labels[i], counters[i], labels[j], counters[j])

    # 전체 공통: 모든 파일에 존재하는 시퀀스
    if n > 2:
        sets = [set(c.keys()) for c in counters]
        universal = sets[0].intersection(*sets[1:])
        union = sets[0].union(*sets[1:])
        print(f'\n{"="*60}')
        print(f'  전체 {n}개 파일 요약')
        print(f'{"="*60}')
        print(f'  모든 파일에 공통으로 존재하는 시퀀스 종류 : {len(universal):,}')
        print(f'  전체 합집합 시퀀스 종류                   : {len(union):,}')
        print(f'  전체 Jaccard (공통/합집합)                : {len(universal)/len(union):.4f}')


def main():
    parser = argparse.ArgumentParser(description='Syscall sequence overlap analysis')
    parser.add_argument('--files', nargs='+', required=True,
                        help='분석할 로그 파일 경로 (2개 이상)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='각 파일의 레이블 (미지정 시 파일명 사용)')
    args = parser.parse_args()

    if len(args.files) < 2:
        print('파일을 2개 이상 지정하세요.')
        return

    labels = args.labels if args.labels else [f.split('/')[-1].replace('.txt', '') for f in args.files]
    if len(labels) != len(args.files):
        print('--labels 개수가 --files 개수와 다릅니다.')
        return

    counters = []
    for label, filepath in zip(labels, args.files):
        print(f'\n[{label}] 파싱 중: {filepath}')
        seqs, counter = extract_sequences(filepath)
        print(f'  총 시퀀스: {len(seqs):,}  /  고유 시퀀스: {len(counter):,}')
        counters.append(counter)

    analyze_all(labels, counters)


if __name__ == '__main__':
    main()
