#!/usr/bin/env bash
# Progress reader for scripts/evaluation/mask_head_generalisation.py.
# The eval prints one row per (generator, class) as it finishes, 10 rows for
# the 5-generator heldout set, so the rate is rows per elapsed second.
# Usage: tools/heldout-progress.sh <log> [--watch]
log=${1:?log path}; shift
# rows = every (generator, class) dir that exists, not a fixed multiple
total=$(ls -d data_sources/heldout/*/*/ 2>/dev/null | grep -vc _masks)
show() {
  [ -f "$log" ] || { echo "no log yet"; return; }
  start=$(stat -c %Y "$log"); now=$(date +%s); el=$((now - start))
  done_=$(grep -cE '^\s+\S+\s+(real|ai_generated|ai_edited)\s+[0-9]+' "$log")
  printf "rows %d/%d   elapsed %dm%02ds" "$done_" "$total" $((el/60)) $((el%60))
  if [ "$done_" -gt 0 ]; then
    rem=$(( (total - done_) * el / done_ ))
    printf "   ETA %dm%02ds (measured rate)" $((rem/60)) $((rem%60))
  else
    printf "   ETA: no rows finished yet, no rate to derive from"
  fi
  if grep -q "saved ->" "$log"; then printf "   DONE"; fi
  if ! pgrep -f "^[^ ]*python scripts/evaluation/mask_head_generalisation" >/dev/null; then printf "   [process not running]"; fi
  echo; tail -n 12 "$log"
}
if [ "${1:-}" = "--watch" ]; then
  while true; do clear; show; grep -q "saved ->" "$log" && break; sleep 15; done
else show; fi
