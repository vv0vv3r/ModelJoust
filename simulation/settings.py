"""
Ustawienia analizy wrażliwości i oceny końcowej: okresy i siatka
parametrów. Jedno miejsce, z którego korzystają run_grid.py i
summarize.py.

Każda symulacja startuje od zera WARMUP_MONTHS przed okresem, który jest
oceniany - rozbieg (pierwszy trening wszystkich modeli, zapełnienie okna
kroczącego MAPE) nie wchodzi do wyników.

UWAGA: import tego modułu (a więc też summarize.py i run_*.py) podstawia
atrapę bazy za db_client (sandbox.py). Nie importować modułów simulation/
w skryptach, które mają pracować na produkcyjnej bazie.
"""
import simulation.sandbox  # noqa: F401  (config.py importuje db_client)

import pandas as pd

from config import DEFAULT_CONFIG

WARMUP_MONTHS = 2

# Okres strojenia - na nim wybierana jest najlepsza kombinacja parametrów.
TUNING_START = "2024-04-15"
TUNING_END = "2024-10-14"

# Okres oceny końcowej - tylko zwycięska kombinacja, rozłączny ze strojeniem.
EVALUATION_START = "2024-10-15"
EVALUATION_END = "2026-10-15"

# Klucze identyczne jak w pipeline_config / DEFAULT_CONFIG. Parametry spoza
# siatki zostają na wartościach z DEFAULT_CONFIG (config.py).
PARAM_GRID = {
    "training_window_years": [0.5, 1, 2],
    "active_model_streak_days": [2, 3],
    "active_model_margin": [0.0025, 0.005],
    "concept_drift_lambda": [0.015, 0.03],
    "rolling_mape_window_days": [14, 29],
}

# Skróty do nazw folderów wyników, np. y0.5_s2_m0.0025_l0.015_w14.
PARAM_CODES = {
    "training_window_years": "y",
    "active_model_streak_days": "s",
    "active_model_margin": "m",
    "concept_drift_lambda": "l",
    "rolling_mape_window_days": "w",
}


def simulation_start(period_start: str) -> str:
    """Pierwszy dzień symulacji = początek ocenianego okresu minus rozbieg."""
    return (pd.Timestamp(period_start) - pd.DateOffset(months=WARMUP_MONTHS)).strftime("%Y-%m-%d")


def build_config(params: dict) -> dict:
    """Pełna konfiguracja symulacji: DEFAULT_CONFIG z nadpisanymi
    parametrami kombinacji."""
    unknown = set(params) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Nieznane parametry (brak w DEFAULT_CONFIG): {unknown}")
    config = {key: value for key, (value, _description) in DEFAULT_CONFIG.items()}
    config.update(params)
    return config


def folder_name(params: dict) -> str:
    return "_".join(f"{PARAM_CODES[key]}{value:g}" for key, value in params.items())
