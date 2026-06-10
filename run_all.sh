#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# Pipeline completa per TUTTE le città in data/data_processed.
#
# Per ogni città:
#   1. Backup dei file stale/vecchi in .bak_TIMESTAMP/  (mv, istantaneo)
#   2. Step 1 (graph + varianti) → job SLURM
#   3. Step 2 (lavoro gravitazionale) → dipende dal suo step 1
#   Alla fine: Supplementary builder → dipende da tutti gli step 2
#
# Uso:
#   bash pipeline_production/run_all.sh           # lancia tutto
#   bash pipeline_production/run_all.sh --dry-run # stampa senza lanciare
# =============================================================================
set -euo pipefail

DRY_RUN=0
OSM_MODE=0
NO_BACKUP=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run"   ]] && DRY_RUN=1
    [[ "$arg" == "--osm"       ]] && OSM_MODE=1
    [[ "$arg" == "--no-backup" ]] && NO_BACKUP=1
done

# Polygon source:
#   default     → GHSL FUA boundary  → graphs_fine_grid/, land/, graphs/
#   --osm       → OSM municipality   → graphs_fine_grid_osm/, land_osm/, graphs_osm/
if [[ "$OSM_MODE" == "1" ]]; then
    POLYGON_SOURCE="osm"
    GRID_SUFFIX="_osm"
    echo "Mode: OSM municipality (--polygon-source osm)"
else
    POLYGON_SOURCE="fua"
    GRID_SUFFIX=""
    echo "Mode: FUA (--polygon-source fua)"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Percorsi ──────────────────────────────────────────────────────────────────
ROOT="/home/fbellisardi/code/topolity"
DATA_ROOT="$ROOT/data/data_processed"
STEP1="$ROOT/pipeline_production/dem_extractor_fine_grid.py"
STEP2="$ROOT/pipeline_production/fine_grid_gravitational_work.py"
SUPP="$ROOT/supplementary/python/build_gravitational_supplementary.py"

MAKE_FIG="$ROOT/pipeline_production/make_figures.py"
FIGURES_ROOT="$ROOT/figures/${POLYGON_SOURCE}"

# ── Risorse SLURM ─────────────────────────────────────────────────────────────
S1_TIME="72:00" ; S1_MEM=128  ; S1_CPU=4
S2_TIME="8:00"  ; S2_MEM=32  ; S2_CPU=4
SB_TIME="2:00"  ; SB_MEM=32  ; SB_CPU=1
FIG_TIME="4:00" ; FIG_MEM=16 ; FIG_CPU=1

# ── Parametri step 1 ─────────────────────────────────────────────────────────
S1_COMMON=(
    --polygon-source "$POLYGON_SOURCE"
    --use-fua
    --step-meters 250
    --num-points 10
    --rotation-angles -40 -30 -20 -15 -10 -5 -1 1 5 10 15 20 30 40
    --ns-scale-factors 1.02 1.05 1.08 1.10 1.12 1.15 1.20
    --ew-scale-factors 1.02 1.05 1.08 1.10 1.12 1.15 1.20
    --land-check full
    --seed 42
    --resume
)

# Angoli di traslazione extra (configurazioni "random" per il notebook)
EXTRA_ANGLES=(--extra-translation-angles 30 45 135 225 315)

# Città che ricevono gli angoli extra
CITIES_WITH_EXTRA=(santiago madrid)

# =============================================================================
# Helper: verifica se una città è nella lista degli angoli extra
# =============================================================================
has_extra_angles() {
    local city="$1"
    for c in "${CITIES_WITH_EXTRA[@]}"; do
        [[ "$c" == "$city" ]] && return 0
    done
    return 1
}

# =============================================================================
# Helper: lancia runlog e restituisce il job ID
# =============================================================================
submit() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] runlog $*" >&2
        echo "$((RANDOM + 10000))"
        return 0
    fi

    local raw
    raw=$(runlog "$@" 2>&1)
    printf '%s\n' "$raw" >&2

    local jid
    jid=$(echo "$raw" | grep -oP '(?i)(?:submitted\s+batch\s+job\s+)\K\d+' | tail -1)
    if [[ -z "$jid" ]]; then
        jid=$(echo "$raw" | grep -oP '\b\d{5,}\b' | tail -1)
    fi
    if [[ -z "$jid" ]]; then
        echo "ERRORE [$label]: impossibile estrarre job ID" >&2
        echo "$raw" >&2
        exit 1
    fi
    echo "$jid"
}

# =============================================================================
# Helper: backup SOURCE-SPECIFIC dei file di una città in .bak_TIMESTAMP/
#
#   Sposta solo le directory del POLYGON_SOURCE corrente:
#     FUA (default) : land/, graphs/, graphs_fine_grid/
#     OSM           : land_osm/, graphs_osm/, graphs_fine_grid_osm/
#   Non tocca mai le directory dell'ALTRO source.
#   Lascia sempre intatti: dem/, data_useful.csv
#
#   Usa --no-backup per saltare completamente (es. re-run su dati già corretti).
# =============================================================================
backup_city() {
    local city="$1"
    local city_dir="$DATA_ROOT/$city"
    local bak_dir="$city_dir/.bak_${TIMESTAMP}"
    local sfx="$GRID_SUFFIX"   # "" for FUA, "_osm" for OSM

    # Check se ci sono dati per questo source
    local needs_backup=0
    [ -d "$city_dir/land${sfx}" ]                           && needs_backup=1
    [ -f "$city_dir/graphs${sfx}/graph_original.pkl" ]      && needs_backup=1
    [ -d "$city_dir/graphs_fine_grid${sfx}" ] && \
        ls "$city_dir/graphs_fine_grid${sfx}"/*.pkl 2>/dev/null | grep -q . \
        && needs_backup=1

    if [[ "$needs_backup" == "0" ]]; then
        return 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY-RUN] backup $city (source=${POLYGON_SOURCE}) → $bak_dir" >&2
        return 0
    fi

    mkdir -p "$bak_dir"

    # Land shapefile + fingerprint (solo per questo source)
    if [ -d "$city_dir/land${sfx}" ]; then
        mv "$city_dir/land${sfx}" "$bak_dir/land${sfx}"
        mkdir -p "$city_dir/land${sfx}"
        echo "  [$city] backed up: land${sfx}/ → .bak_${TIMESTAMP}/"
    fi

    # Grafo originale (solo per questo source)
    if [ -f "$city_dir/graphs${sfx}/graph_original.pkl" ]; then
        mkdir -p "$bak_dir/graphs${sfx}"
        mv "$city_dir/graphs${sfx}/graph_original.pkl" "$bak_dir/graphs${sfx}/"
        echo "  [$city] backed up: graphs${sfx}/graph_original.pkl"
    fi

    # Fine-grid: pkls + stats + work CSV (solo per questo source)
    if [ -d "$city_dir/graphs_fine_grid${sfx}" ]; then
        mv "$city_dir/graphs_fine_grid${sfx}" "$bak_dir/graphs_fine_grid${sfx}"
        mkdir -p "$city_dir/graphs_fine_grid${sfx}"
        echo "  [$city] backed up: graphs_fine_grid${sfx}/ → .bak_${TIMESTAMP}/"
    fi
}

# =============================================================================
# Auto-discovery delle città: tutte le cartelle con data_useful.csv
# =============================================================================
mapfile -t ALL_CITIES < <(
    for d in "$DATA_ROOT"/*/; do
        [ -f "$d/data_useful.csv" ] && basename "$d"
    done | sort
)

echo ""
echo "════════════════════════════════════════════════════"
echo " run_all.sh — ${#ALL_CITIES[@]} città trovate in data_processed"
echo " Source: $POLYGON_SOURCE  |  Backup: $([ $NO_BACKUP -eq 1 ] && echo SKIP || echo $TIMESTAMP)"
echo " Dry-run: $DRY_RUN"
echo "════════════════════════════════════════════════════"
echo ""

# =============================================================================
# 0. Backup SOURCE-SPECIFIC (skippato con --no-backup)
#    Muove solo le directory del source corrente; non tocca l'altro source.
# =============================================================================
echo "─── [0/4] Backup (source=${POLYGON_SOURCE}) ─────────────────"
if [[ "$NO_BACKUP" == "1" ]]; then
    echo "  --no-backup: skip backup, riuso i dati esistenti"
else
for city in "${ALL_CITIES[@]}"; do
    backup_city "$city"
done
echo "  Backup completato ($TIMESTAMP)"
fi   # end of backup block (skipped with --no-backup)
echo ""

# =============================================================================
# 1. Step 1 — tutte le città in parallelo
# =============================================================================
# Job name prefix per evitare clash quando FUA e OSM girano insieme
JOB_PFX="${POLYGON_SOURCE}_"   # "fua_" oppure "osm_"

echo "─── [1/4] Step 1 — graph extraction (${POLYGON_SOURCE}) ──────"
declare -A S1_IDS

for city in "${ALL_CITIES[@]}"; do
    jname="${JOB_PFX}fg_${city}"
    if has_extra_angles "$city"; then
        jid=$(submit "$jname" \
            -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "$jname" \
            python -u "$STEP1" --city "$city" "${S1_COMMON[@]}" "${EXTRA_ANGLES[@]}")
        echo "  $jname → job $jid  [+ extra angles]"
    else
        jid=$(submit "$jname" \
            -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "$jname" \
            python -u "$STEP1" --city "$city" "${S1_COMMON[@]}")
        echo "  $jname → job $jid"
    fi
    S1_IDS[$city]=$jid
done
echo ""

# =============================================================================
# 2. Step 2 — ogni città parte solo dopo il suo step 1 (-d JOB_ID)
# =============================================================================
echo "─── [2/4] Step 2 — gravitational work (${POLYGON_SOURCE}) ────"
declare -A S2_IDS

for city in "${ALL_CITIES[@]}"; do
    s1_jid="${S1_IDS[$city]}"
    jname="${JOB_PFX}fgw_${city}"
    jid=$(submit "$jname" \
        -t "$S2_TIME" -m "$S2_MEM" -c "$S2_CPU" -j "$jname" \
        -d "$s1_jid" \
        python "$STEP2" \
            --data-root "$DATA_ROOT" \
            --city "$city" \
            --polygon-source "$POLYGON_SOURCE" \
            --resume)
    S2_IDS[$city]=$jid
    echo "  $jname → job $jid  (after $s1_jid)"
done
echo ""

# # Raccoglie tutti gli step2 IDs per le dipendenze successive
# all_s2_ids=$(printf '%s:' "${S2_IDS[@]}")
# all_s2_ids="${all_s2_ids%:}"

# # =============================================================================
# # 3. Figure del paper — un job per città (dipende dal suo step 2)
# #    Genera: fig2 (DEM map), fig4 (sensitivity), fig5a (bar), fig5b (boxplot)
# #    Risorse: 16 GB RAM (DEM raster clippato) + 4 h (sequenziale, leggero)
# # =============================================================================
# echo "─── [3/5] Figure del paper ─────────────────────────"

# declare -A FIG_IDS

# for city in "${ALL_CITIES[@]}"; do
#     s2_jid="${S2_IDS[$city]}"
#     jid=$(submit "fig_${city}" \
#         -t "$FIG_TIME" -m "$FIG_MEM" -c "$FIG_CPU" -j "fig_${city}" \
#         -d "$s2_jid" \
#         python "$MAKE_FIG" \
#             --city "$city" \
#             --polygon-source "$POLYGON_SOURCE" \
#             --output-root "$FIGURES_ROOT")
#     FIG_IDS[$city]=$jid
#     echo "  fig_${city} → job $jid  (after step2 $s2_jid)"
# done
# echo ""

# # =============================================================================
# # 4. Job globale — world map (Fig. 2a) + supplementary builder
# #    Dipende da tutti gli step 2 (non da figure per-città).
# # =============================================================================
# echo "─── [4/5] World map + Supplementary ───────────────"

# all_fig_ids=$(printf '%s:' "${FIG_IDS[@]}")
# all_fig_ids="${all_fig_ids%:}"

# # Supplementary builder dipende da tutti step2
# sb_jid=$(submit "supp_builder" \
#     -t "$SB_TIME" -m "$SB_MEM" -c "$SB_CPU" -j "supp_builder" \
#     -d "$all_s2_ids" \
#     python "$SUPP" --only all --output-root "$FIGURES_ROOT/supplementary")
# echo "  supp_builder → job $sb_jid  (after all step2)"

# # World map + figure globali dipendono da tutti i job figura per-città
# wm_jid=$(submit "fig_global" \
#     -t "2:00" -m 16 -c 1 -j "fig_global" \
#     -d "$all_fig_ids" \
#     python "$MAKE_FIG" \
#         --world-map-only \
#         --output-root "$FIGURES_ROOT")
# echo "  fig_global (world map) → job $wm_jid  (after all fig_* jobs)"
# echo ""

# # =============================================================================
# # 5. Riepilogo
# # =============================================================================
# echo "─── [5/5] Riepilogo ────────────────────────────────"
# echo ""
# printf "  %-20s %-10s %-10s %-10s\n" "CITTÀ" "STEP1" "STEP2" "FIGURE"
# for city in "${ALL_CITIES[@]}"; do
#     printf "  %-20s %-10s %-10s %-10s\n" \
#         "$city" "${S1_IDS[$city]}" "${S2_IDS[$city]}" "${FIG_IDS[$city]}"
# done
# echo ""
# echo "  SUPPLEMENTARY: $sb_jid"
# echo "  WORLD MAP:     $wm_jid"
# echo ""
# echo "  Monitoraggio:"
# echo "    watch -n 30 'squeue -u \$USER --format=\"%.8i %.18j %.8T %.10M %R\"'"
# echo ""
# echo "  Figure salvate in: $FIGURES_ROOT/{city}/"
# echo "    fig2_{city}.png/pdf   — DEM elevazione + rete stradale + confine FUA"
# echo "    fig4_{city}.png/pdf   — Energia vs parametro trasformazione (copia step2)"
# echo "    fig5a_{city}.png/pdf  — Bar chart: originale vs trasformazioni"
# echo "    fig5b_{city}.png/pdf  — Boxplot: distribuzione energia per tipo"
# echo "    fua_map_{city}.png    — Mappa FUA con basemap CartoDB (da step1)"
# echo "    global/fig2a_world_map.png/pdf — Mappa mondo tutte le città"
# echo ""
# echo "  Backup in: \$DATA_ROOT/<city>/.bak_${TIMESTAMP}/"
# echo "  Per ripristinare una città:"
# echo "    cd \$DATA_ROOT/<city> && mv .bak_${TIMESTAMP}/land land \\"
# echo "      && mv .bak_${TIMESTAMP}/graphs_fine_grid graphs_fine_grid \\"
# echo "      && mv .bak_${TIMESTAMP}/graphs/graph_original.pkl graphs/"
# echo ""
# echo "  Fallback manuale se dipendenze multiple non supportate:"
# echo "    python $MAKE_FIG --all --output-root $FIGURES_ROOT"
# echo "    python $SUPP --only all"
