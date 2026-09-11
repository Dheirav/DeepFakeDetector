#!/usr/bin/env bash
# Progress reader for scripts/training/train_mask_head.py logs.
# The trainer prints "batch N/M" every 200 batches and one "epoch E/T" line
# per epoch, so total work is M*T batches and the rate is measured from the
# batches completed since the log was created.
# Usage: tools/train-progress.sh <log> [--watch]
log=${1:?log path}; shift
show() {
  [ -f "$log" ] || { echo "no log yet"; return; }
  start=$(stat -c %W "$log" 2>/dev/null); [ "$start" = 0 ] && start=$(stat -c %Y "$log")
  now=$(date +%s); el=$((now - start))
  per=$(grep -oE 'batch [0-9]+/[0-9]+' "$log" | tail -1 | sed 's|.*/||')
  tot_ep=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$log" | tail -1 | sed 's|.*/||')
  ep_done=$(grep -cE '^\s+epoch +[0-9]+/[0-9]+ ' "$log")
  last_b=$(grep -oE 'batch [0-9]+/[0-9]+' "$log" | tail -1 | sed 's|batch ||;s|/.*||')
  # batches printed after the last epoch line belong to the current epoch
  tail_after=$(awk '/^\s+epoch +[0-9]+\//{n=NR} END{print n+0}' "$log")
  cur_b=$(awk -v n="$tail_after" 'NR>n && /batch [0-9]+\//{b=$0} END{if(b){sub(/.*batch /,"",b);sub(/\/.*/,"",b);print b}else print 0}' "$log")
  if [ -z "$per" ] || [ -z "$tot_ep" ]; then
    printf "elapsed %dm%02ds   no batch counter yet (encoder loading)\n" $((el/60)) $((el%60))
  else
    done_b=$((ep_done * per + cur_b)); total_b=$((tot_ep * per))
    printf "epoch %d/%d   batches %d/%d   elapsed %dm%02ds" "$ep_done" "$tot_ep" "$done_b" "$total_b" $((el/60)) $((el%60))
    if [ "$done_b" -gt 0 ]; then
      rem=$(( (total_b - done_b) * el / done_b ))
      printf "   ETA %dh%02dm (measured rate, includes per-epoch eval)" $((rem/3600)) $(((rem%3600)/60))
    fi
    echo
  fi
  grep -q "saved ->" "$log" && echo "DONE"
  pgrep -f "^[^ ]*python scripts/training/train_mask_head" >/dev/null || echo "[process not running]"
  grep -E "^\s+epoch|Traceback|Error|Killed" "$log" | tail -n 12
}
if [ "${1:-}" = "--watch" ]; then
  while true; do clear; show; grep -q "saved ->" "$log" && break; sleep 30; done
else show; fi
