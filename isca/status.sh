#!/bin/bash
# One-screen overview of the ISCA runs: jobs (named m-<profile>) and sessions done per model.
#   bash isca/status.sh
cd "$(dirname "$0")/.."
squeue -u "$USER" -o "%.9i %.16j %.2t %.10M %.12R" | sed 's/^/  /'
echo
printf "  %-14s %7s %7s\n" model done target
for p in qwen35_27b local qwen35_122b gpt_oss_120b llama4_scout gemma4_31b llama31_8b gemma3_12b granite42_8b; do
  n=$(cat data/jobs/queue-$p-*/worker*.log 2>/dev/null | grep -c " ok ")
  case $p in qwen35_27b|local) t=1782 ;; *) t=504 ;; esac
  printf "  %-14s %7s %7s\n" "$p" "$n" "$t"
done
echo
echo "  GPU start-up failures in the last 10 jobs:"
for f in $(ls -t data/jobs/isca-*.log 2>/dev/null | head -10); do
  r=$(grep -hE "GPU found|could not use" "$f" | tail -1)
  printf "    %-28s %-14s %s\n" "$(basename "$f")" "$(head -1 "$f" | grep -oE 'profile=[a-z0-9_]+' | cut -d= -f2)" "$r"
done
