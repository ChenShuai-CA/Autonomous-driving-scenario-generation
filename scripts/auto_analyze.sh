#!/usr/bin/env bash
# 自动周期分析脚本：监控 autoscale_log.csv 的 epoch 进度并按间隔运行分析
# 使用方法：
#   bash scripts/auto_analyze.sh \
#       --log log_rebound_full_run1/autoscale_log.csv \
#       --out log_rebound_full_run1/autoscale_analysis.txt \
#       --interval 5 \
#       --python-env vae \
#       --sleep 120
# 依赖：scripts/analyze_autoscale.py 已存在。
# 逻辑：
#   1. 记录上次分析的 epoch 行数 (忽略 CSV 头，如果没有头则直接用行数)
#   2. 循环：检测当前行数换算 epoch，如果增量 >= interval 则调用分析脚本追加结果
#   3. 支持中途 ctrl+c 安全退出

set -euo pipefail

LOG_FILE=""
OUT_FILE=""
INTERVAL=5
SLEEP_SEC=120
PY_ENV=""
PY_CMD="python"

function usage() {
  grep '^#' "$0" | sed 's/^# //'
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) LOG_FILE="$2"; shift 2;;
    --out) OUT_FILE="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --sleep) SLEEP_SEC="$2"; shift 2;;
    --python-env) PY_ENV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

if [[ -z "$LOG_FILE" ]]; then
  echo "必须指定 --log autoscale_log.csv"; exit 1
fi
if [[ -z "$OUT_FILE" ]]; then
  OUT_FILE="${LOG_FILE%.csv}_analysis.txt"
fi

if [[ -n "$PY_ENV" ]]; then
  # 尝试激活 conda env（假设 anaconda 安装路径）
  if [[ -f ~/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate "$PY_ENV" || { echo "激活环境失败: $PY_ENV"; exit 1; }
    PY_CMD="python"
  else
    echo "警告: 未找到 conda.sh，使用系统 python" >&2
  fi
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "等待日志文件创建: $LOG_FILE"
  while [[ ! -f "$LOG_FILE" ]]; do
    sleep 5
  done
fi

mkdir -p "$(dirname "$OUT_FILE")"

echo "==== auto_analyze 启动 $(date '+%F %T') ====" | tee -a "$OUT_FILE"
echo "log=$LOG_FILE interval=$INTERVAL sleep=$SLEEP_SEC env=$PY_ENV" | tee -a "$OUT_FILE"

LAST_EPOCH=-1
trap 'echo "捕获中断, 退出." | tee -a "$OUT_FILE"; exit 0' INT TERM

while true; do
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "文件消失，等待重建..." | tee -a "$OUT_FILE"
    sleep "$SLEEP_SEC"
    continue
  fi
  # 行数 -> epoch (假设每行一个 epoch；若文件含 header 可在此适配)
  LINE_CNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
  CURRENT_EPOCH=$(( LINE_CNT ))
  # 若第一行为数据而非 header 则 epoch=行数；如果未来添加 header 可减 1

  if (( LAST_EPOCH < 0 )); then
    LAST_EPOCH=$CURRENT_EPOCH
    echo "初始检测: epoch=$CURRENT_EPOCH" | tee -a "$OUT_FILE"
  else
    DELTA=$(( CURRENT_EPOCH - LAST_EPOCH ))
    if (( DELTA >= INTERVAL )); then
      echo "[触发] $(date '+%F %T') 检测到新增 $DELTA epoch (from $LAST_EPOCH to $CURRENT_EPOCH) -> 运行分析" | tee -a "$OUT_FILE"
      { 
        echo "--- 分析开始 epoch=$CURRENT_EPOCH 时间=$(date '+%F %T') ---";
        $PY_CMD scripts/analyze_autoscale.py --log "$LOG_FILE" || echo "分析脚本返回非零状态";
        echo "--- 分析结束 epoch=$CURRENT_EPOCH ---";
      } >> "$OUT_FILE" 2>&1
      LAST_EPOCH=$CURRENT_EPOCH
    fi
  fi
  sleep "$SLEEP_SEC"
done
