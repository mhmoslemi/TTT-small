#!/usr/bin/env bash
# make_explorer.sh - build the interactive HTML explorer for one (or more) run directories.
#
#   ./make_explorer.sh ../runs/ac1_Qwen3.8-27B_1108-1201-step3            # -> out/<run name>.html
#   ./make_explorer.sh ../runs/ac1_Qwen3.8-27B_1108-1201-step3 --open      # ... and open it in the browser
#   ./make_explorer.sh                                                     # newest run under ../runs
#   ./make_explorer.sh RUN_A RUN_B --all-code                              # several runs, full source embedded
#   ./make_explorer.sh --unified RUN_AC1 RUN_ERDOS --open                  # ONE html with all problems (landing page + every explorer)
#   ./make_explorer.sh --unified                                           # ... using the newest run of every problem under ../runs
#
# Read-only on the run directory and on the rest of the code base; writes only into ./out (or -o).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [RUN_DIR ...] [options]

  RUN_DIR              run directory (contains config.json and stepNN/ folders); quote it if the name has spaces.
                       Omit to use the most recently modified run under ../runs.

Options:
  -o, --out FILE       output html path (only with a single RUN_DIR or with --unified; default: out/<run name>.html)
  --unified            build ONE html holding all the given runs (different problems): a landing page that compares them
                       (results vs published records, GPU specification, wall clock) plus the full explorer of each run.
                       Without RUN_DIRs the runs listed in unified_runs.txt are used. Default: out/all_problems.html
  --open               open the result in the default browser when done
  --all-code           embed the source of every valid program (bigger file, ~23 MB for 12 steps)
  --code-per-step N    embed source for the N best valid programs of each step (default 25)
  -h, --help           show this help

Environment:
  PYTHON               python interpreter to use (default: python3, needs >= 3.8)
EOF
}

RUNS=()
EXTRA=()
OUT=""
OPEN=0
UNIFIED=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -o|--out) [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }; OUT="$2"; shift 2 ;;
    --open) OPEN=1; shift ;;
    --unified) UNIFIED=1; shift ;;
    --all-code) EXTRA+=(--all-code); shift ;;
    --code-per-step) [ $# -ge 2 ] || { echo "error: $1 needs a value" >&2; exit 2; }; EXTRA+=(--code-per-step "$2"); shift 2 ;;
    -*) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) RUNS+=("$1"); shift ;;
  esac
done

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: '$PY' not found - install Python 3.8+ or set PYTHON=/path/to/python3" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
  echo "error: $PY is older than 3.8" >&2
  exit 1
fi
if [ "$UNIFIED" -eq 0 ] && [ -n "$OUT" ] && [ "${#RUNS[@]}" -gt 1 ]; then
  echo "error: -o/--out can only be used with a single run directory" >&2
  exit 2
fi

build_one() {
  local run="$1" out="$2"
  if [ -n "$run" ]; then
    if [ ! -d "$run" ]; then echo "error: not a directory: $run" >&2; return 1; fi
    run="$(cd "$run" && pwd)"
    if [ ! -f "$run/config.json" ] && ! ls -d "$run"/step[0-9]* >/dev/null 2>&1; then
      echo "error: '$run' does not look like a run directory (no config.json / stepNN folders)" >&2
      return 1
    fi
  fi
  local args=()
  [ -n "$run" ] && args+=("$run")
  [ -n "$out" ] && args+=(-o "$out")
  # shellcheck disable=SC2068
  "$PY" "$HERE/build_explorer.py" ${args[@]+"${args[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} || return 1
  if [ -z "$out" ]; then
    if [ -n "$run" ]; then out="$HERE/out/$(basename "$run").html"; else out=""; fi
  fi
  LAST_OUT="$out"
}

build_unified() {
  local args=(--unified) r
  for r in ${RUNS[@]+"${RUNS[@]}"}; do
    if [ ! -d "$r" ]; then echo "error: not a directory: $r" >&2; return 1; fi
    args+=("$(cd "$r" && pwd)")
  done
  [ -n "$OUT" ] && args+=(-o "$OUT")
  "$PY" "$HERE/build_explorer.py" "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"} || return 1
  LAST_OUT="${OUT:-$HERE/out/all_problems.html}"
}

LAST_OUT=""
STATUS=0
if [ "$UNIFIED" -eq 1 ]; then
  build_unified || STATUS=1
elif [ "${#RUNS[@]}" -eq 0 ]; then
  build_one "" "$OUT" || STATUS=1
else
  for r in "${RUNS[@]}"; do
    build_one "$r" "$OUT" || STATUS=1
  done
fi

if [ "$STATUS" -eq 0 ] && [ "$OPEN" -eq 1 ] && [ -n "$LAST_OUT" ] && [ -f "$LAST_OUT" ]; then
  if command -v open >/dev/null 2>&1; then open "$LAST_OUT"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$LAST_OUT" >/dev/null 2>&1 &
  else echo "(no 'open' / 'xdg-open' found - open the file manually)"; fi
fi
exit "$STATUS"
