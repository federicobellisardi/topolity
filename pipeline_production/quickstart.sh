#!/bin/bash
# QUICKSTART - Topolity Pipeline Production
# Copia-incolla questi comandi per iniziare subito

echo "======================================================================"
echo "TOPOLITY FINE-GRID ANALYSIS - QUICKSTART GUIDE"
echo "======================================================================"

# Step 0: Naviga alla cartella
cd /home/fbellisardi/code/topolity/pipeline_production

# Step 1: Validazione rapida (1 minuto)
echo ""
echo "[1/4] Validazione pipeline..."
bash validate_pipeline.sh

# Step 2: TEST veloce (modalità test su Santiago, ~10-30 minuti)
echo ""
echo "[2/4] Generando grafi TEST per Santiago (parametri ridotti)..."
conda run -n geo_flow python dem_extractor_fine_grid.py \
  --city santiago \
  --test \
  --resume

# Step 3: Calcolo lavoro TEST
echo ""
echo "[3/4] Calcolando lavoro gravitazionale per Santiago..."
conda run -n geo_flow python fine_grid_gravitational_work.py \
  --city santiago \
  --resume

# Step 4: Verificare risultati
echo ""
echo "[4/4] Controllando risultati..."
echo ""
echo "Risultati Santiago:"
ls -lh /home/fbellisardi/code/topolity/data/data_processed/santiago/graphs_fine_grid/*.csv 2>/dev/null || echo "  (CSV non ancora disponibili)"
echo ""

# Success message
echo "======================================================================"
echo "✓ QUICKSTART COMPLETATO"
echo "======================================================================"
echo ""
echo "PROSSIMI STEP:"
echo ""
echo "1. Lanciare generazione grafi per MULTIPLE CITTÀ:"
echo "   python dem_extractor_fine_grid.py --city madrid --resume"
echo "   python dem_extractor_fine_grid.py --city barcelona --resume"
echo "   python dem_extractor_fine_grid.py --city sevilla --resume"
echo ""
echo "2. Calcolo lavoro per TUTTE le città insieme:"
echo "   python fine_grid_gravitational_work.py --cities santiago madrid barcelona sevilla --resume"
echo ""
echo "3. Analisi risultati (con parametri REALI):"
echo "   python dem_extractor_fine_grid.py --city santiago"
echo "     --step-meters 500 --num-points 5"
echo "     --rotation-angles -20 -15 -10 -5 5 10 15 20"
echo "     --ns-scale-factors 0.9 0.95 1.0 1.05 1.1"
echo "     --ew-scale-factors 0.9 0.95 1.0 1.05 1.1"
echo "     --workers 8"
echo ""
echo "Per documentazione completa:"
echo "   cat README_PIPELINE.md"
echo ""
