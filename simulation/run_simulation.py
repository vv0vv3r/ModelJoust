"""
Jedna symulacja systemu: main_pipeline.main() dzień po dniu na danych
historycznych, z bazą w pamięci (fake_db) zamiast Supabase.

Kod systemu działa bez zmian - podmienione są tylko połączenia z
usługami zewnętrznymi (sandbox.py) i pobieranie danych (dane są już
wgrane z simulation/data/market_data.csv).

Wynik w simulation/results/<nazwa>/:
  models_logs.csv, model_predictions.csv, system_logs.csv,
  pipeline_config.csv - tabele jak w bazie, pełna historia działania,
  pipeline.log        - wszystko, co system wypisał,
  meta.json           - parametry, okres, czas trwania, błędy dni.
meta.json zapisywany jest jako ostatni - jego obecność oznacza, że
symulacja doszła do końca.

Uruchomienie (z katalogu repo), np.:
  python -m simulation.run_simulation --name test --start 2024-04-01 --end 2024-04-12 \
      --param data_drift_z_threshold=2.5 --param training_window_years=5
"""
import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

import simulation.sandbox as sandbox  # atrapy przed importem kodu systemu

import pandas as pd

import main_pipeline
from config import DEFAULT_CONFIG

SIM_DIR = Path(__file__).resolve().parent
MARKET_DATA_PATH = SIM_DIR / "data" / "market_data.csv"
RESULTS_DIR = SIM_DIR / "results"

# Dane są wgrane z pliku na starcie symulacji - pobieranie z yfinance w
# każdym dniu jest zbędne.
main_pipeline.save_latest_market_data = lambda as_of_date: None

_market_data_loaded = False


def _ensure_market_data() -> None:
    global _market_data_loaded
    if not _market_data_loaded:
        market_df = pd.read_csv(MARKET_DATA_PATH, index_col="target_date")
        sandbox.fake_supabase.load_raw_data(market_df)
        _market_data_loaded = True


def _seed_config(params: dict) -> dict:
    """pipeline_config = DEFAULT_CONFIG z nadpisanymi parametrami
    testowanej kombinacji."""
    unknown = set(params) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Nieznane parametry (brak w DEFAULT_CONFIG): {unknown}")
    config = {key: value for key, (value, _description) in DEFAULT_CONFIG.items()}
    config.update(params)
    sandbox.fake_supabase.table("pipeline_config").insert([
        {"key": key, "value": value, "description": DEFAULT_CONFIG[key][1], "active": True}
        for key, value in config.items()
    ]).execute()
    return config


def run_simulation(name: str, start_date: str, end_date: str, params: dict | None = None) -> Path:
    """Symuluje dni robocze od start_date do end_date włącznie i zapisuje
    wyniki do RESULTS_DIR/name. Zwraca ścieżkę folderu wyników."""
    params = params or {}
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_market_data()
    sandbox.fake_supabase.reset_results()
    config = _seed_config(params)

    sim_dates = pd.bdate_range(start=start_date, end=end_date)
    print(f"[{name}] {len(sim_dates)} dni: {sim_dates[0].date()} -> {sim_dates[-1].date()}, parametry: {params}")

    failures = []
    t0 = time.time()
    with open(out_dir / "pipeline.log", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        for i, date in enumerate(sim_dates, start=1):
            print(f"\n{'=' * 80}\nDzień {i}/{len(sim_dates)}: {date.date()}\n{'=' * 80}")
            try:
                main_pipeline.main(as_of_date=date)
            except Exception as e:
                traceback.print_exc(file=log)
                failures.append({"date": str(date.date()), "error": repr(e)})
            if i % 20 == 0 or i == len(sim_dates):
                with contextlib.redirect_stdout(sys.__stdout__):
                    print(f"[{name}] dzień {i}/{len(sim_dates)} ({(time.time() - t0) / 60:.1f} min)")

    sandbox.fake_supabase.dump_tables(out_dir)
    meta = {
        "name": name,
        "start_date": start_date,
        "end_date": end_date,
        "params": params,
        "config": config,
        "sim_days": len(sim_dates),
        "runtime_minutes": round((time.time() - t0) / 60, 2),
        "failures": failures,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{name}] gotowe w {meta['runtime_minutes']} min, błędów dni: {len(failures)} -> {out_dir}")
    return out_dir


def _parse_param(text: str) -> tuple[str, object]:
    key, _, raw = text.partition("=")
    return key, json.loads(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jedna symulacja systemu ModelJoust na danych historycznych.")
    parser.add_argument("--name", required=True, help="nazwa folderu wyników w simulation/results/")
    parser.add_argument("--start", required=True, help="pierwszy dzień symulacji (RRRR-MM-DD)")
    parser.add_argument("--end", required=True, help="ostatni dzień symulacji (RRRR-MM-DD)")
    parser.add_argument("--param", action="append", default=[], type=_parse_param,
                        help="nadpisanie parametru pipeline_config, np. data_drift_z_threshold=2.5")
    args = parser.parse_args()
    run_simulation(args.name, args.start, args.end, dict(args.param))
