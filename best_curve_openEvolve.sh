# sh best_curve_openEvolve.sh /work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs/openevolve_20260611_184912.log 512


#!/bin/sh
# Parse an OpenEvolve circle_packing log into a best-so-far sum_radii curve.
# Usage:
#   sh best_curve.sh                 # newest log in the default logs dir
#   sh best_curve.sh /path/to.log    # a specific log
#   sh best_curve.sh /path/to.log 256  # custom step size (default 512)
set -eu

LOG_DIR="/work/mohammad/TTT-local/openevolve/examples/circle_packing/openevolve_output/logs"
STEP_SIZE="${2:-512}"

if [ -n "${1:-}" ]; then
  LOG="$1"
else
  LOG="$(ls -t "$LOG_DIR"/openevolve_*.log 2>/dev/null | head -n 1 || true)"
fi

[ -n "${LOG:-}" ] && [ -f "$LOG" ] || { echo "log file not found (pass it as arg 1)" >&2; exit 1; }

OUT_DIR="$(dirname "$LOG")"
BASE="$(basename "$LOG" .log)"
OUT_JSON="${OUT_DIR}/${BASE}_best_curve.json"
OUT_STEP="${OUT_DIR}/${BASE}_step_summary.tsv"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# extract (iteration, sum_radii) pairs, then sort by iteration number
awk '
  /- Iteration [0-9]+:.*completed/ {
    for (i=1;i<=NF;i++) if ($i=="Iteration") { c=$(i+1); sub(/:$/,"",c); h=1; break }
    next
  }
  h && /- Metrics:/ {
    for (i=1;i<=NF;i++) if ($i ~ /^sum_radii=/) {
      v=$i; sub(/^sum_radii=/,"",v); sub(/,$/,"",v); print c"\t"v; h=0; break
    }
  }
' "$LOG" | sort -n -k1,1 > "$TMP"

[ -s "$TMP" ] || { echo "no iterations parsed from $LOG" >&2; exit 1; }

# detailed best-so-far curve (JSON)
{
  echo "["
  awk -v step="$STEP_SIZE" '
    { it=$1; sr=$2+0; if (sr>best) best=sr; s=int(it/step);
      printf "%s  {\"iteration\": %d, \"sum_radii\": %.4f, \"best_so_far\": %.4f, \"step\": %d}", (NR>1?",\n":""), it, sr, best, s }
    END { print "" }
  ' "$TMP"
  echo "]"
} > "$OUT_JSON"

# per-step summary: cumulative best sum_radii at the end of each step
awk -v step="$STEP_SIZE" '
  BEGIN { print "step\tlast_iteration\tbest_sum_radii" }
  { it=$1; sr=$2+0; if (sr>best) best=sr; s=int(it/step);
    sb[s]=best; if (it>li[s]) li[s]=it; if (s>ms) ms=s }
  END { for (k=0;k<=ms;k++) if (k in sb) printf "%d\t%d\t%.4f\n", k, li[k], sb[k] }
' "$TMP" > "$OUT_STEP"

n="$(wc -l < "$TMP" | tr -d ' ')"
overall_best="$(awk -F'\t' '{if($2>b)b=$2} END{printf "%.4f", b}' "$TMP")"
echo "parsed $n iterations from $(basename "$LOG")"
echo "overall best sum_radii: $overall_best"
echo "wrote $OUT_JSON"
echo "wrote $OUT_STEP"
echo
echo "step summary:"
cat "$OUT_STEP"