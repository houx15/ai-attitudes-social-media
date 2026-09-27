#!/usr/bin/env bash
# Start Stage 1 (LLM labelling) for one platform as NUM_TASKS parallel background tasks.
# Usage: scripts/start_stage1.sh {weibo|twitter} [NUM_TASKS]   (default 4)
# Uses the `python` of the active environment, so activate it first, e.g.
#   conda activate opinion && ./run_twitter.sh
# (or set PYTHON=/path/to/python). Requests in flight ~= NUM_TASKS x MAX_WORKERS
# (config.py). Rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."

PLATFORM="${1:?usage: start_stage1.sh weibo|twitter [NUM_TASKS]}"
NUM_TASKS="${2:-4}"
case "$PLATFORM" in weibo|twitter) ;; *) echo "platform must be weibo or twitter" >&2; exit 1 ;; esac

running="$(pgrep -f "run_analysis.py $PLATFORM" || true)"
if [ -n "$running" ]; then
  echo "Stage 1 for $PLATFORM is already running (pids: $(echo $running))." >&2
  echo "Stop it first: pkill -f 'run_analysis.py $PLATFORM'" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" > /dev/null; then
  echo "'$PYTHON' not found; activate your environment first (e.g. conda activate opinion)" >&2
  exit 1
fi
echo "Using $(command -v "$PYTHON") ($("$PYTHON" --version 2>&1))"

mkdir -p logs
for i in $(seq 1 "$NUM_TASKS"); do
  log="logs/${PLATFORM}_task${i}of${NUM_TASKS}.log"
  nohup "$PYTHON" run_analysis.py "$PLATFORM" --task_id "$i" --num_tasks "$NUM_TASKS" \
    > "$log" 2>&1 &
  echo "started $PLATFORM task $i/$NUM_TASKS (pid $!) -> $log"
done

echo
echo "Watch: tail -f logs/${PLATFORM}_task*of${NUM_TASKS}.log"
echo "Stop:  pkill -f 'run_analysis.py $PLATFORM'   (labels so far are saved; rerun to resume)"
