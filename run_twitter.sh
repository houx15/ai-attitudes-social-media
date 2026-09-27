#!/usr/bin/env bash
# Stage 1 for Twitter. Usage: ./run_twitter.sh [NUM_TASKS]   (default 4)
exec "$(dirname "$0")/scripts/start_stage1.sh" twitter "$@"
