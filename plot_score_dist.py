"""
Anomaly score distribution visualization for model drift motivation figure.

eval_save.py --save_scores 로 저장한 .npy 파일을 로드해 boxplot / violin plot 생성.

파일 명명 규칙 (eval_save.py가 자동 생성):
    {scores_dir}/{version}_{service}_benign.npy
    {scores_dir}/{version}_{service}_attack.npy

Usage:
    python plot_score_dist.py --scores_dir ./scores/adservice --attack_label Attack

    # 레이블 순서 직접 지정 (기본: 파일명 알파벳 순)
    python plot_score_dist.py \
        --scores_dir ./scores/adservice \
        --order "v0.3.6_(train)" v0.4.0 v0.5.0 v0.6.0 v0.7.0 v0.8.0 v0.9.0 v0.10.5 Attack
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# threshold: 기준 버전(v0.3.6) benign의 95th percentile
# --threshold 인자로 덮어쓸 수 있음
DEFAULT_THRESHOLD = None  # None이면 첫 번째 benign의 95th를 자동 사용

# 색상 + hatch: 흑백 인쇄에서도 구분되도록 설계
# 항목 수에 따라 순서대로 할당됨
_PALETTE = [
    dict(color='#FFFFFF', hatch=''),          # 기준 (흰색)
    dict(color='#C6DBEF', hatch='....'),
    dict(color='#9ECAE1', hatch='////'),
    dict(color='#6BAED6', hatch='xxxx'),
    dict(color='#4292C6', hatch='----'),
    dict(color='#2171B5', hatch='\\\\\\\\'),
    dict(color='#084594', hatch='++++'),
    dict(color='#525252', hatch='||||'),
]
_ATTACK_STYLE = dict(color='#000000', hatch='')  # attack: 검정


def load_scores(scores_dir, order=None, attack_label='Attack'):
    """
    scores_dir에서 *_benign.npy / *_attack.npy 로드.
    반환: OrderedDict { label: np.ndarray }
    attack은 마지막에 하나로 합쳐서 추가.
    """
    benign_files = sorted(
        f for f in os.listdir(scores_dir) if f.endswith('_benign.npy')
    )

    # label 추출: {version}_{service}_benign.npy → {version}_{service}
    def to_label(fname):
        return fname.replace('_benign.npy', '')

    if order:
        # 사용자 지정 순서
        label_map = {to_label(f): f for f in benign_files}
        benign_files = [label_map[o] for o in order if o in label_map]

    samples = {}
    for fname in benign_files:
        label = to_label(fname)
        samples[label] = np.load(os.path.join(scores_dir, fname))
        print(f'  Loaded benign [{label}]: {len(samples[label])} samples')

    # attack: 같은 디렉토리의 *_attack.npy 전부 합치기
    attack_files = sorted(
        f for f in os.listdir(scores_dir) if f.endswith('_attack.npy')
    )
    if attack_files:
        attack_arrays = [np.load(os.path.join(scores_dir, f)) for f in attack_files]
        samples[attack_label] = np.concatenate(attack_arrays)
        print(f'  Loaded attack [{attack_label}]: {len(samples[attack_label])} samples '
              f'(from {len(attack_files)} file(s))')

    return samples


def assign_styles(labels, attack_label):
    styles = {}
    benign_labels = [l for l in labels if l != attack_label]
    for i, label in enumerate(benign_labels):
        styles[label] = _PALETTE[i % len(_PALETTE)]
    if attack_label in labels:
        styles[attack_label] = _ATTACK_STYLE
    return styles


def _setup_ax(ax, labels, threshold, threshold_label):
    ax.axhline(threshold, color='#444444', linestyle='--', linewidth=1.2,
               label=f'Threshold ({threshold_label} 95th = {threshold:.2f})', zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Anomaly Score', fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.legend(fontsize=9)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_boxplot(samples, styles, threshold, threshold_label, save_path):
    labels = list(samples.keys())
    data = [samples[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5))

    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.0, color='#333333'),
                    capprops=dict(linewidth=1.0, color='#333333'),
                    boxprops=dict(linewidth=1.0))

    for patch, label in zip(bp['boxes'], labels):
        s = styles[label]
        patch.set_facecolor(s['color'])
        patch.set_hatch(s['hatch'])
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    _setup_ax(ax, labels, threshold, threshold_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def plot_violin(samples, styles, threshold, threshold_label, save_path):
    labels = list(samples.keys())
    data = [samples[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5))

    parts = ax.violinplot(data, positions=range(1, len(labels) + 1),
                          showmedians=True, showextrema=True)

    for pc, label in zip(parts['bodies'], labels):
        s = styles[label]
        pc.set_facecolor(s['color'])
        pc.set_hatch(s['hatch'])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
        pc.set_alpha(0.9)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2.0)
    parts['cmaxes'].set_color('#555555')
    parts['cmins'].set_color('#555555')
    parts['cbars'].set_color('#555555')
    parts['cbars'].set_linewidth(0.8)

    _setup_ax(ax, labels, threshold, threshold_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot anomaly score distributions')
    parser.add_argument('--scores_dir', type=str, required=True,
                        help='eval_save.py --save_scores 로 저장한 .npy 파일 디렉토리')
    parser.add_argument('--attack_label', type=str, default='Attack',
                        help='attack 항목 레이블 (기본: Attack)')
    parser.add_argument('--order', type=str, nargs='*', default=None,
                        help='레이블 표시 순서 (미지정 시 알파벳 순)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='threshold 값 (미지정 시 첫 번째 benign의 95th percentile 자동 사용)')
    parser.add_argument('--threshold_label', type=str, default=None,
                        help='threshold 설명에 쓸 레이블 (미지정 시 첫 번째 benign 레이블 사용)')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='결과 figure 저장 디렉토리 (기본: 현재 디렉토리)')
    args = parser.parse_args()

    print(f'Loading scores from {args.scores_dir} ...')
    samples = load_scores(args.scores_dir, args.order, args.attack_label)

    labels = list(samples.keys())
    styles = assign_styles(labels, args.attack_label)

    # threshold 결정
    benign_labels = [l for l in labels if l != args.attack_label]
    threshold_label = args.threshold_label or benign_labels[0]
    threshold = args.threshold or float(np.percentile(samples[benign_labels[0]], 95))
    print(f'Threshold: {threshold:.4f} ({threshold_label} 95th)')

    os.makedirs(args.out_dir, exist_ok=True)
    plot_boxplot(samples, styles, threshold, threshold_label,
                 os.path.join(args.out_dir, 'fig_score_boxplot.pdf'))
    plot_violin(samples, styles, threshold, threshold_label,
                os.path.join(args.out_dir, 'fig_score_violin.pdf'))


if __name__ == '__main__':
    main()
