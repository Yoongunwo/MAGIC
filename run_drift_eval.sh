#!/usr/bin/env bash
# run_drift_eval.sh
#
# v0.3.6 모델을 기준으로 v0.3.6 ~ v0.10.5 버전 전체에 대해 eval_save.py를 실행.
#
# Usage:
#   bash run_drift_eval.sh <service> [--attack_dir <path>] [options]
#
# Examples:
#   bash run_drift_eval.sh cartservice
#   bash run_drift_eval.sh frontend --attack_dir ./data/Attack
#   bash run_drift_eval.sh cartservice --device 0 --syscall_dim 449
#
# 추가 인자는 eval_save.py에 그대로 전달됩니다.
# (--device, --syscall_dim, --window_size, --stride, --n_neighbors, 등)

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_drift_eval.sh <service> [eval_save.py options...]"
    echo "Example: bash run_drift_eval.sh cartservice --device 0"
    exit 1
fi

SERVICE="$1"
shift  # 나머지 인자들은 eval_save.py로 전달

VERSIONS=(v0.3.6 v0.4.0 v0.5.0 v0.6.0 v0.7.0 v0.8.0 v0.9.0 v0.10.5)
REF_VERSION="v0.3.6"

DATA_BASE="./data/online_boutique"
CHECKPOINT="./checkpoints/online_boutique/${REF_VERSION}/${SERVICE}.pt"
SCORES_BASE="./scores/online_boutique/${SERVICE}/${REF_VERSION}"
ATTACK_DIR="./data/Attack"

echo "========================================"
echo " Service   : ${SERVICE}"
echo " Ref model : ${CHECKPOINT}"
echo " Scores →  : ${SCORES_BASE}/"
echo " Versions  : ${VERSIONS[*]}"
echo "========================================"

for VERSION in "${VERSIONS[@]}"; do
    BENIGN_PATH="${DATA_BASE}/${VERSION}/${SERVICE}.txt"
    BENIGN_CACHE="${DATA_BASE}/${VERSION}/${SERVICE}.pkl"

    if [ ! -f "${BENIGN_PATH}" ]; then
        echo ""
        echo "[SKIP] ${VERSION}: ${BENIGN_PATH} 없음"
        continue
    fi

    echo ""
    echo "──────────────────────────────────────"
    echo " ${VERSION}"
    echo "──────────────────────────────────────"

    # v0.3.6은 ref == test이므로 ref_path 불필요 (기본 train_ratio 모드)
    if [ "${VERSION}" = "${REF_VERSION}" ]; then
        python eval_save.py \
            --benign_path  "${BENIGN_PATH}" \
            --benign_cache "${BENIGN_CACHE}" \
            --checkpoint   "${CHECKPOINT}" \
            --attack_paths "${ATTACK_DIR}" \
            --save_scores  "${SCORES_BASE}" \
            "$@"
    else
        REF_PATH="${DATA_BASE}/${REF_VERSION}/${SERVICE}.txt"
        REF_CACHE="${DATA_BASE}/${REF_VERSION}/${SERVICE}.pkl"
        python eval_save.py \
            --benign_path  "${BENIGN_PATH}" \
            --benign_cache "${BENIGN_CACHE}" \
            --ref_path     "${REF_PATH}" \
            --ref_cache    "${REF_CACHE}" \
            --checkpoint   "${CHECKPOINT}" \
            --attack_paths "${ATTACK_DIR}" \
            --save_scores  "${SCORES_BASE}" \
            "$@"
    fi
done

echo ""
echo "========================================"
echo " Done. Scores saved to: ${SCORES_BASE}/"
echo "========================================"
