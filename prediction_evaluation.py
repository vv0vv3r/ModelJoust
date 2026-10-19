import pandas as pd

from db_client import supabase

MODEL_TYPES = ["naive", "OLS", "lasso", "random_forest", "xgboost"]


def beats_baseline(mape_by_model: dict, challenger: str, baseline: str, margin: float) -> bool:
    # Współdzielone przez dzisiejsze porównanie i przeliczanie historii streaka
    # zawsze względem JEDNEGO, tego samego baseline'u.
    baseline_mape = (mape_by_model or {}).get(baseline)
    challenger_mape = (mape_by_model or {}).get(challenger)
    if baseline_mape is None or challenger_mape is None or baseline_mape == 0:
        return False
    return ((baseline_mape - challenger_mape) / baseline_mape) >= margin

def get_all_model_ids() -> dict:
    """Zwraca {model_id: model_type} dla WSZYSTKICH wersji w models_logs
    (nie tylko aktywnych) - potrzebne do poprawnego mapowania historycznych
    predykcji sprzed retreningu przy liczeniu kroczącego MAPE.

    Stronicowanie przez .range() - Supabase zwraca najwyżej 1000 wierszy na
    zapytanie; bez tego po 1000 wersjach modeli część z nich nie miałaby
    typu i ich predykcje cicho wypadałyby z MAPE i Page-Hinkleya."""
    page_size = 1000
    rows = []
    start = 0
    while True:
        response = (
            supabase.table("models_logs")
            .select("id, model_type")
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows.extend(response.data)
        if len(response.data) < page_size:
            break
        start += page_size

    if not rows:
        # Legalny stan startowy (pierwsze uruchomienie systemu, jeszcze
        # zero modeli w models_logs) - nie błąd. Nie ma żadnych predykcji do
        # zmapowania, więc reszta funkcji bezpiecznie przejdzie z pustym dict.
        print("Brak jakichkolwiek wpisów w models_logs - zakładam pierwsze uruchomienie systemu.")
        return {}
    return {row["id"]: row["model_type"] for row in rows}


def evaluate_predictions_and_update_system_logs(today_str: str, config: dict):
    """
    Ewaluuje wczorajsze predykcje ("pending") względem dzisiejszej
    rzeczywistej ceny złota, liczy kroczące MAPE per model w oknie
    rolling_mape_window_days, sprawdza czy któryś model niebędący aktywnym
    pobija aktywny przez active_model_streak_days dni z rzędu (i ewentualnie
    przełącza aktywny model), i zapisuje wynik dnia do system_logs.

    Zwraca {model_type: dzisiejszy_błąd_procentowy} dla modeli, które miały
    dziś zaewaluowaną predykcję - wejście do detektora Concept Drift
    (Page-Hinkley) w main_pipeline.py.
    """
    models = list(MODEL_TYPES)
    id_to_type = get_all_model_ids()

    raw_response = (
        supabase.table("raw_data")
        .select("actual_y")
        .eq("target_date", today_str)
        .execute()
    )

    if not raw_response.data:
        print(f"Brak wiersza w raw_data dla {today_str} - pomijam cały krok ewaluacji.")
        return {}

    actual_gold_value = raw_response.data[0]["actual_y"]

    pending_response = (
        supabase.table("model_predictions")
        .select("prediction_id, model_id, predicted_value")
        .eq("target_date", today_str)
        .eq("status", "pending")
        .execute()
    )

    todays_errors = {}

    # --- Krok 1: ewaluacja oczekujących predykcji. Brak notowania złota ->
    # wszystkie oznaczone jako 'unused'. Jest notowanie -> liczony błąd,
    # zapis do model_predictions i wypełnienie todays_errors per model. ---
    if pending_response.data:
        if actual_gold_value is None:
            for pred in pending_response.data:
                supabase.table("model_predictions").update(
                    {"status": "unused"}
                ).eq("prediction_id", pred["prediction_id"]).execute()
            print(f"Złoto bez notowania na {today_str} - oznaczono "
                  f"{len(pending_response.data)} predykcji jako 'unused'.")
        else:
            for pred in pending_response.data:
                error_value = pred["predicted_value"] - actual_gold_value
                percentage_error = abs(error_value) / abs(actual_gold_value)

                supabase.table("model_predictions").update({
                    "actual_value": actual_gold_value,
                    "error_value": error_value,
                    "status": "evaluated",
                }).eq("prediction_id", pred["prediction_id"]).execute()

                model_type = id_to_type.get(pred["model_id"])
                if model_type:
                    todays_errors[model_type] = percentage_error
            print(f"Zaewaluowano {len(pending_response.data)} predykcji na {today_str}.")
    else:
        print(f"Brak oczekujących predykcji na {today_str}.")

    # --- Krok 2: kroczące MAPE per model_type, na podstawie evaluated.
    # rolling_mape_window_days z configu to "-1" względem realnej długości
    # okna: gte/lte są obustronnie domknięte, więc "dziś" + (N-1) dni wstecz
    # daje faktycznie N-dniowe okno. ---
    window_start = (pd.Timestamp(today_str) - pd.Timedelta(days=int(config["rolling_mape_window_days"]))).strftime("%Y-%m-%d")

    evaluated_response = (
        supabase.table("model_predictions")
        .select("model_id, predicted_value, actual_value")
        .eq("status", "evaluated")
        .gte("target_date", window_start)
        .lte("target_date", today_str)
        .execute()
    )

    evaluated_df = pd.DataFrame(evaluated_response.data)
    if not evaluated_df.empty:
        evaluated_df["model_type"] = evaluated_df["model_id"].map(id_to_type)

    today_mapes = {}
    for model_type in models:
        if evaluated_df.empty:
            today_mapes[model_type] = None
            continue
        model_rows = evaluated_df[evaluated_df["model_type"] == model_type]
        if model_rows.empty:
            today_mapes[model_type] = None
            continue
        ape = (model_rows["predicted_value"] - model_rows["actual_value"]).abs() / model_rows["actual_value"].abs()
        today_mapes[model_type] = float(ape.mean() * 100)

    # --- Krok 3: ustalenie aktywnego modelu ---
    streak_days = int(config["active_model_streak_days"])

    last_logs_response = (
        supabase.table("system_logs")
        .select("*")
        .order("log_date", desc=True)
        # Co najmniej 1 wiersz - z niego bierze się dzisiejszy aktywny model
        # (przy streak_days=1 limit(0) nie zwróciłby nic i aktywny byłby
        # zawsze naiwny). Nadmiarowy wiersz odcina niżej [-streak_days:].
        .limit(max(streak_days - 1, 1))
        .execute()
    )
    last_logs = list(reversed(last_logs_response.data))  # od najstarszego do najnowszego

    if last_logs:
        current_active = last_logs[-1]["active_model"]
    else:
        current_active = "naive"  # pierwsze uruchomienie systemu - start od modelu bazowego

    today_flags = {}
    for model_type in models:
        if model_type == current_active:
            today_flags[model_type] = None
            continue
        today_flags[model_type] = beats_baseline(
            today_mapes, model_type, current_active, config["active_model_margin"]
        )

    new_active_model = current_active
    qualifying_challengers = []

    for model_type in models:
        if model_type == current_active:
            continue
        # Historia PRZELICZANA na bieżąco względem dzisiejszego current_active
        # (przez zapisane per-dnia mape wszystkich modeli), nie odczytywana z
        # zapisanego beats_active - ten był liczony względem modelu aktywnego
        # W TAMTYM dniu, który mógł być inny niż dzisiejszy current_active,
        # gdyby aktywny model zmienił się w trakcie okna streak_days.
        history_flags = [
            beats_baseline(row.get("mape") or {}, model_type, current_active, config["active_model_margin"])
            for row in last_logs
        ] + [today_flags[model_type]]
        last_window = history_flags[-streak_days:]
        if len(last_window) == streak_days and all(v is True for v in last_window):
            qualifying_challengers.append(model_type)

    if qualifying_challengers:
        # Jeśli więcej niż jeden pretendent spełnia warunek tego samego dnia,
        # wygrywa ten z niższym dzisiejszym MAPE
        new_active_model = min(qualifying_challengers, key=lambda m: today_mapes[m])

    # --- Krok 4: zapis wiersza do system_logs ---
    payload = {
        "log_date": today_str,
        "active_model": new_active_model,
        "mape": {model_type: today_mapes.get(model_type) for model_type in models},
        "beats_active": {model_type: today_flags.get(model_type) for model_type in models},
    }

    try:
        supabase.table("system_logs").upsert(payload, on_conflict="log_date").execute()
        print(f"Zapisano system_logs dla {today_str}. Aktywny model: {new_active_model} "
              f"(MAPE: {today_mapes}).")
    except Exception as e:
        print(f"Błąd zapisu system_logs: {e}")

    return todays_errors