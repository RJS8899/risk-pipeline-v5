# Risk Pipeline (Complete)

Robuste, wöchentliche Country-Risk-Pipeline mit mehreren Datenquellen (World Bank, WGI, OWID-Spiegel von UNODC/UNHCR/UCDP/EM-DAT, ND-GAIN, optional ACLED/GPI).
- **Outputs**: `out/risk_wide.csv`, `out/risk_long.csv`, `out/meta.json`
- **Kein Crash bei Ausfällen**: Quellen haben Fallbacks; fehlende Werte werden imputiert/umgewichtet.
- **0..100-Normierung**: Höher = riskanter.

## Lokaler Run
```bash
pip install -e .
risk-pipeline
```
Ergebnisse in `out/`.

## GitHub Actions
Workflow liegt in `.github/workflows/weekly.yml`. Manuell starten via **Actions → Run workflow**.

## Optionale Env-Variablen
- `ACLED_EMAIL`, `ACLED_KEY` – wenn vorhanden, werden ACLED-Fatalities pro 100k genutzt, sonst fallback/umgewichtung.

## Vendor-CSV (optional)
Lege zusätzliche CSVs in `data/vendor/` ab – Schema: `iso3,year,value`.
- `unodc_robbery.csv`
- `gpi_scores.csv`

Wenn diese fehlen, läuft der Build trotzdem; betroffene Indikator-Gewichte werden automatisch umverteilt.
