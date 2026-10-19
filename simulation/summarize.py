"""
Podsumowanie symulacji na podstawie zapisanych CSV (tabel z bazy w
pamięci) - jeden wiersz na symulację, liczony tylko z ocenianego okresu
(bez rozbiegu).

"System" = prognoza modelu, który był aktywny w chwili jej tworzenia.
Prognozę na dzień D system tworzy w przebiegu poprzedniej sesji, po
wyborze aktywnego modelu - więc dla D liczy się active_model z
OSTATNIEGO wiersza system_logs sprzed D (nie z wiersza dnia D, który
powstaje dopiero po poznaniu ceny z D).

Trafność kierunku liczona względem ostatniej znanej ceny w chwili
prognozy, czyli prognozy modelu naiwnego z tego samego dnia: czy model
dobrze wskazał, czy cena wzrośnie, czy spadnie. Dni, w których prognoza
równa się ostatniej cenie (np. aktywny model naiwny - "brak zdania" o
kierunku), i dni bez zmiany ceny są pomijane.
"""
import json
from pathlib import Path

import pandas as pd

from simulation.settings import (
    PARAM_GRID, TUNING_END, TUNING_START, build_config, folder_name, simulation_start,
)

MODEL_TYPES = ["naive", "OLS", "lasso", "random_forest", "xgboost"]
MODEL_CODES = {"naive": "n", "OLS": "o", "lasso": "l", "random_forest": "r", "xgboost": "x"}


def load_daily_predictions(folder: Path, period_start: str, period_end: str) -> pd.DataFrame:
    """Ocenione prognozy z okresu w układzie dzień x model (kolumny =
    prognozy każdego modelu i systemu) + cena rzeczywista i aktywny model.
    Ta sama tabela posłuży do testów (Diebold-Mariano, MCS,
    Pesaran-Timmermann) w ocenie końcowej."""
    models_logs = pd.read_csv(folder / "models_logs.csv")
    predictions = pd.read_csv(folder / "model_predictions.csv")
    system_logs = pd.read_csv(folder / "system_logs.csv")

    predictions = predictions[predictions["status"] == "evaluated"].copy()
    predictions["model_type"] = predictions["model_id"].map(models_logs.set_index("id")["model_type"])
    predictions = predictions[predictions["target_date"].between(period_start, period_end)]

    daily = predictions.pivot(index="target_date", columns="model_type", values="predicted_value")
    daily["actual"] = predictions.groupby("target_date")["actual_value"].first()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    logs = system_logs[["log_date", "active_model"]].copy()
    logs["log_date"] = pd.to_datetime(logs["log_date"])
    daily = pd.merge_asof(
        daily.reset_index(), logs.sort_values("log_date"),
        left_on="target_date", right_on="log_date", allow_exact_matches=False,
    ).set_index("target_date").drop(columns="log_date")

    daily["system"] = [
        row[row["active_model"]] if pd.notna(row["active_model"]) else None
        for _, row in daily.iterrows()
    ]
    return daily


def _metrics(predicted: pd.Series, actual: pd.Series, last_price: pd.Series) -> tuple[float, float, int]:
    """(MAPE w %, trafność kierunku w %, liczba dni bez zdania o kierunku).
    Dni bez prognozy pomijane."""
    has_prediction = predicted.notna()
    predicted, actual, last_price = predicted[has_prediction], actual[has_prediction], last_price[has_prediction]
    ape = (predicted - actual).abs() / actual.abs()
    no_opinion = predicted == last_price
    counted = ~no_opinion & (actual != last_price)
    direction_hit = ((predicted - last_price) > 0) == ((actual - last_price) > 0)
    direction = float(direction_hit[counted].mean() * 100) if counted.any() else float("nan")
    return float(ape.mean() * 100), direction, int(no_opinion.sum())


def summarize_run(folder: Path, period_start: str, period_end: str) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    daily = load_daily_predictions(folder, period_start, period_end)

    summary = {"folder": folder.name, **meta["params"]}

    for column in ["system"] + MODEL_TYPES:
        mape, direction, no_opinion_days = _metrics(daily[column], daily["actual"], daily["naive"])
        summary[f"mape_{column}"] = round(mape, 4)
        summary[f"kierunek_{column}"] = round(direction, 2) if column != "naive" else None
        if column == "system":
            summary["dni_bez_kierunku_system"] = no_opinion_days
    summary["system_vs_naiwny_proc"] = round(
        (summary["mape_system"] - summary["mape_naive"]) / summary["mape_naive"] * 100, 2
    )

    sequence = "".join(MODEL_CODES[m] for m in daily["active_model"])
    summary["zmiany_championa"] = sum(a != b for a, b in zip(sequence, sequence[1:]))

    # Dzień retreningu = sesja po train_end_date (model trenuje na danych do
    # poprzedniej sesji, bo cel to cena z kolejnego dnia).
    models_logs = pd.read_csv(folder / "models_logs.csv")
    retrain_day = (
        pd.to_datetime(models_logs["train_end_date"]) + pd.tseries.offsets.BDay(1)
    ).dt.strftime("%Y-%m-%d")
    in_period = models_logs[
        retrain_day.between(period_start, period_end)
        & (models_logs["retrain_trigger"] != "init")
    ]
    triggers = in_period["retrain_trigger"].fillna("")
    summary["retreningi"] = len(in_period)
    summary["retreningi_DD"] = int(triggers.str.startswith("DD").sum())
    summary["retreningi_CD"] = int((triggers == "CD").sum())

    summary["dni_oceny"] = len(daily)
    summary["bledy_dni"] = len(meta["failures"])
    summary["czas_min"] = meta["runtime_minutes"]
    summary["commit"] = (meta.get("commit") or "")[:7]
    summary["niezacommitowane_zmiany"] = meta.get("uncommitted_changes")
    summary["sekwencja_championa"] = sequence
    return summary


def grid_run_problem(folder: Path) -> str | None:
    """None, jeśli folder to zakończona symulacja siatki zgodna z BIEŻĄCYMI
    ustawieniami (nazwa wg folder_name, okres strojenia z rozbiegiem, pełna
    konfiguracja). W przeciwnym razie powód - np. folder z poprzednich
    ustawień albo nie z siatki (test, ocena końcowa)."""
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return "brak meta.json (symulacja niezakończona)"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    params = meta["params"]
    if set(params) != set(PARAM_GRID) or folder.name != folder_name(params):
        return "nie jest folderem siatki"
    if meta["start_date"] != simulation_start(TUNING_START) or meta["end_date"] != TUNING_END:
        return f"inny okres ({meta['start_date']} -> {meta['end_date']})"
    if meta["config"] != build_config(params):
        return "inna konfiguracja parametrów spoza siatki"
    return None


def build_grid_summary(results_dir: Path) -> pd.DataFrame:
    """Podsumowanie wszystkich zakończonych symulacji siatki zgodnych z
    bieżącymi ustawieniami (grid_run_problem), posortowane od najlepszego
    MAPE systemu."""
    rows = []
    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir() and grid_run_problem(folder) is None:
            rows.append(summarize_run(folder, TUNING_START, TUNING_END))
    df = pd.DataFrame(rows)
    return df.sort_values("mape_system").reset_index(drop=True) if not df.empty else df
