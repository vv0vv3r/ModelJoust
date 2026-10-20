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
TUNING_START = "2023-10-15"
TUNING_END = "2024-10-14"

# Okres oceny końcowej - tylko zwycięska kombinacja, rozłączny ze strojeniem.
EVALUATION_START = "2024-10-15"
EVALUATION_END = "2026-10-15"

# Klucze identyczne jak w pipeline_config / DEFAULT_CONFIG. Parametry spoza
# siatki zostają na wartościach z DEFAULT_CONFIG (config.py).
#
# Parametry z jedną wartością (ich wartością domyślną) są w siatce, żeby
# od początku były częścią nazwy folderu - dopisanie im później kolejnej
# wartości policzy tylko nowe kombinacje, a dotychczasowe zostaną użyte.
PARAM_GRID = {
    "training_window_years": [0.5, 1, 1.5, 2],
    "active_model_streak_days": [2, 3],
    "active_model_margin": [0.0025, 0.005],
    "concept_drift_lambda": [0.015, 0.03],
    "rolling_mape_window_days": [14, 21, 29],
    "data_drift_z_threshold": [2.5, 3.0, 4.0],
    "data_drift_pct_threshold": [0.2],
    "concept_drift_delta": [0.0005],
    "tree_cumulative_importance_threshold": [0.8, 0.9],
}

# Skróty do nazw folderów wyników, np. y0.5_s2_m0.0025_l0.015_w14_z3_p0.2_d0.0005_t0.9.
PARAM_CODES = {
    "training_window_years": "y",
    "active_model_streak_days": "s",
    "active_model_margin": "m",
    "concept_drift_lambda": "l",
    "rolling_mape_window_days": "w",
    "data_drift_z_threshold": "z",
    "data_drift_pct_threshold": "p",
    "concept_drift_delta": "d",
    "tree_cumulative_importance_threshold": "t",
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
