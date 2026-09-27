#!/usr/bin/env bash
# Stage 1 for Weibo. Usage: ./run_weibo.sh [NUM_TASKS]   (default 4)
exec "$(dirname "$0")/scripts/start_stage1.sh" weibo "$@"
