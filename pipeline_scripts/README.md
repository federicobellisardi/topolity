# Workflow Scripts

Questo folder contiene gli script necessari per il workflow di analisi.

## Elenco Script
- **dem_extractor_fine_grid.py**: Estrazione DEM su grid fine.
- **gravitational_work.py**: Calcolo del lavoro gravitazionale.
- **fine_grid_gravitational_work.py**: Analisi fine-grid del lavoro gravitazionale.
- **transformation.py**: Trasformazioni di coordinate e grafi.
- **map_generator.py**: Generatore di mappe e visualizzazioni.

## Workflow in 3 Passi
1. **Generazione grafi trasformati**: Utilizzare `transformation.py` e `map_generator.py` per processare i dati iniziali.
2. **Calcolo lavoro gravitazionale**: Eseguire `gravitational_work.py` per calcolare i potenziali e il lavoro sulle traiettorie.
3. **Analisi/Plot fine-grid**: Utilizzare `dem_extractor_fine_grid.py` e `fine_grid_gravitational_work.py` per l'analisi di dettaglio e la visualizzazione finale.
