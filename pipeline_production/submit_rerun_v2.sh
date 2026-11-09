#!/usr/bin/env bash
set -euo pipefail

# Round-2 rerun (method_version 2): per city, archive the previous fine-grid
# outputs, then submit step-1 -> step-2 (and densification for its cities)
# as a dependency chain. Run on a node where Slurm (`sbatch`, used by `run`)
# is available:
#
#   DRY_RUN=1 bash pipeline_production/submit_rerun_v2.sh                 # print only
#   DRY_RUN=0 CITIES="barcelone" bash pipeline_production/submit_rerun_v2.sh
#   DRY_RUN=0 bash pipeline_production/submit_rerun_v2.sh                 # all cities
#
# Previous outputs are moved to <dir>_v1, never deleted; if a step-1
# submission fails, that city's archive is moved back.

# Repository root: TOPOLITY_ROOT, else the parent of this script, else the Slurm submission directory
# (sbatch runs a copy of job scripts from its spool directory; the submit scripts cd to the root first).
_topolity_root() {
  local d
  for d in "${TOPOLITY_ROOT:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
    [[ -n "$d" && -f "$d/python/study_config.py" ]] && { echo "$d"; return 0; }
  done
  echo "cannot locate the topolity repository: set TOPOLITY_ROOT" >&2; return 1
}
ROOT="$(_topolity_root)"
# Machine-specific settings (PY, RUNLOG), not versioned: see pipeline_production/local.env.example
if [[ -f "$ROOT/pipeline_production/local.env" ]]; then source "$ROOT/pipeline_production/local.env"; fi
DATA_ROOT="$ROOT/data/data_processed"
PY="${PY:-python}"
RUNLOG="${RUNLOG:-runlog}"
STEP1="$ROOT/pipeline_production/dem_extractor_fine_grid.py"
STEP2="$ROOT/pipeline_production/fine_grid_gravitational_work.py"
DENSIF="$ROOT/python/multi_city_settlement.py"
DENSIF_OUT="$ROOT/output/terrain_aware_densification"
ARCHIVE_SUFFIX="${ARCHIVE_SUFFIX:-_v1}"
DENSIF_CITIES="milan barcelone toronto chicago amsterdam bandung bruxelles bogota"

T1="${T1:-300:00}"; M1="${M1:-128}"; C1="${C1:-4}"
T2="${T2:-120:00}"; M2="${M2:-128}"; C2="${C2:-4}"
TD="${TD:-48:00}";  MD="${MD:-64}";  CD="${CD:-1}"
OD_SAMPLES="${OD_SAMPLES:-5}"

# Variant grid of the submitted results (run_all.sh), so that the rerun covers
# exactly the configurations in graphs_fine_grid_v1.
STEP1_GRID=(
  --step-meters 250 --num-points 10
  --rotation-angles -40 -30 -20 -15 -10 -5 -1 1 5 10 15 20 30 40
)
EXPANSIONS="1.02 1.05 1.08 1.10 1.12 1.15 1.20"
CONTRACTIONS="0.80 0.85 0.88 0.90 0.92 0.95 0.98"
# Cities with extra translation directions and contractions (added in a later run)
EXTRA_CITIES="santiago madrid"
EXTRA_ANGLES="30 45 135 225 315"
DRY_RUN="${DRY_RUN:-1}"        # 1=print only, 0=archive + submit
CITIES="${CITIES:-}"           # optional space-separated subset
EXCLUDE_CITIES="${EXCLUDE_CITIES:-pekin shanghai}"   # removed from the study (China)
RESUME_ONLY="${RESUME_ONLY:-0}"  # 1 = do not archive; continue the current graphs_fine_grid (e.g. after OOM)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$ROOT"

# ---- preflight: fail before touching any directory -------------------------
for f in "$PY" "$RUNLOG" "$STEP1" "$STEP2" "$DENSIF"; do
  [[ -e "$f" ]] || { echo "[abort] missing $f" >&2; exit 1; }
done
if [[ "$DRY_RUN" != "1" ]]; then
  command -v run >/dev/null || { echo "[abort] 'run' not in PATH on $(hostname)" >&2; exit 1; }
  command -v sbatch >/dev/null || { echo "[abort] 'sbatch' not in PATH on $(hostname): use a Slurm submit node" >&2; exit 1; }
fi

submit() {  # prints the job id, empty on failure
  local out
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  runlog $*" >&2
    echo "DRYRUN"
    return 0
  fi
  out="$("$RUNLOG" "$@" 2>&1)" || true
  echo "$out" >&2
  sed -nE 's/^Submitted batch job ([0-9]+).*/\1/p' <<<"$out" | tail -n1
}

archive() {  # sets ARCHIVED=1 if something was moved (no subshell: exit must stop the script)
  local src="$1" dst="$1$ARCHIVE_SUFFIX"
  ARCHIVED=0
  [[ -e "$src" ]] || return 0
  # an empty directory left by an interrupted launch holds nothing to archive
  if [[ -d "$src" && -z "$(ls -A "$src")" ]]; then
    return 0
  fi
  if [[ -e "$dst" ]]; then
    echo "[abort] $dst already exists; refusing to overwrite an archive" >&2
    exit 1
  fi
  echo "  mv $src -> $dst" >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    mv "$src" "$dst"
    ARCHIVED=1
  fi
}

restore() {
  local src="$1" dst="$1$ARCHIVE_SUFFIX"
  if [[ -e "$dst" && ! -e "$src" ]]; then
    mv "$dst" "$src"
    echo "  restored $src" >&2
  fi
}

if [[ -z "$CITIES" ]]; then
  CITIES="$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)"
fi

densif_archived=0
submitted=() failed=()

for city in $CITIES; do
  [[ -f "$DATA_ROOT/$city/data_useful.csv" ]] || continue
  if [[ " $EXCLUDE_CITIES " == *" $city "* ]]; then
    echo "[skip] $city (excluded)" >&2
    continue
  fi
  fg_dir="$DATA_ROOT/$city/graphs_fine_grid"
  ARCHIVED=0
  if [[ "$RESUME_ONLY" == "1" ]]; then
    echo "[$city] resume only (no archive)" >&2
  # archive present and no new outputs yet: this city was already submitted
  elif [[ -e "$fg_dir$ARCHIVE_SUFFIX" ]] && { [[ ! -e "$fg_dir" ]] || [[ -z "$(ls -A "$fg_dir")" ]]; }; then
    echo "[skip] $city (already submitted: $fg_dir$ARCHIVE_SUFFIX exists)" >&2
    continue
  else
    echo "[$city]" >&2
    archive "$fg_dir"
  fi

  scales="$EXPANSIONS"
  extra=()
  if [[ " $EXTRA_CITIES " == *" $city "* ]]; then
    scales="$EXPANSIONS $CONTRACTIONS"
    extra=(--extra-translation-angles $EXTRA_ANGLES)
  fi
  # shellcheck disable=SC2086
  j1="$(submit -t "$T1" -m "$M1" -c "$C1" -j "fg2_${city}" -K fg2_step1 \
        "$PY" -u "$STEP1" --base-path "$DATA_ROOT" --city "$city" \
        "${STEP1_GRID[@]}" --ns-scale-factors $scales --ew-scale-factors $scales "${extra[@]}" \
        --workers 1 --seed 42 --resume --dem-mode tree --low-memory --use-fua --land-check full)"
  if [[ -z "$j1" ]]; then
    echo "[error] step-1 submission failed for $city" >&2
    [[ "$ARCHIVED" == "1" ]] && restore "$fg_dir"
    failed+=("$city")
    continue
  fi

  j2="$(submit -t "$T2" -m "$M2" -c "$C2" -j "fgw2_${city}" -K fgw2_step2 -d "$j1" \
        "$PY" -u "$STEP2" --data-root "$DATA_ROOT" --city "$city" --resume --low-memory \
        --od-samples "$OD_SAMPLES")"
  [[ -n "$j2" ]] || { echo "[error] step-2 submission failed for $city (step-1 job $j1 is queued)" >&2; failed+=("$city:step2"); }

  if [[ " $DENSIF_CITIES " == *" $city "* ]]; then
    if [[ "$densif_archived" == "0" ]]; then
      archive "$DENSIF_OUT"
      [[ "$DRY_RUN" == "1" ]] || mkdir -p "$DENSIF_OUT"
      densif_archived=1
    fi
    jd="$(submit -t "$TD" -m "$MD" -c "$CD" -j "densif2_${city}" -K densif2 -d "$j1" \
          "$PY" -u "$DENSIF" --city "$city")"
    [[ -n "$jd" ]] || { echo "[error] densification submission failed for $city" >&2; failed+=("$city:densif"); }
  fi
  submitted+=("$city")
done

echo "Submitted: ${#submitted[@]} cities" >&2
if (( ${#failed[@]} )); then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "Done." >&2
